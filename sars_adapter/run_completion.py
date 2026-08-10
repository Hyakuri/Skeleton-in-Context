"""Skeleton-in-Context 的 SARS-Inter V2 query 直接运行入口。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import subprocess
import time

import numpy as np
import torch

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from sars_adapter.checkpoint_identity import (
    resolve_checkpoint_source,
    sha256_file as _identity_sha256_file,
    validate_completion_checkpoint_identity,
)
from sars_adapter.contracts import load_completion_query
from sars_adapter.coordinate_adapter import TRANSFORM_MODE
from sars_adapter.inference import (
    build_train_prompt_pool,
    build_train_prompt_provider,
    complete_f64_with_model,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_PICKLE_PROTOCOL = 4
OFFICIAL_UPSTREAM_REPOSITORY_URL = (
    'https://github.com/fanglaosi/Skeleton-in-Context'
)
OFFICIAL_UPSTREAM_COMMIT = '361e1c0b9552baa8510e00dbb33629debfd66873'
ADAPTER_FORK_REPOSITORY_URL = 'https://github.com/Hyakuri/Skeleton-in-Context'


def build_direct_run_config():
    """集中放置外部补全推理需要手动调整的参数。"""
    return {
        'checkpoint_source_mode': 'identity_manifest',  # 正式使用身份清单；smoke 可改为 direct_path。
        'checkpoint_identity_manifest_path': '<SIC_CHECKPOINT_IDENTITY_JSON>',  # 正式 bundle 中的身份清单。
        'dry_run': True,  # True=仅检查输入；False=执行真实 GPU 推理。
        'query_path': '<COMPLETION_QUERY_PATH>',  # SARS-Inter V2 query，不得传评价 sidecar。
        'checkpoint_path': '<SIC_CHECKPOINT_PATH>',  # 仅 direct_path 冒烟模式填写；程序自动计算 SHA256。
        'source_config': '<SIC_EFFECTIVE_CONFIG_PATH>',  # 仅 direct_path 填写，且必须与 checkpoint 来自同一次训练。
        'checkpoint_identity_policy': 'require_mc_only',  # identity_manifest 固定 require_mc_only；allow_legacy 只允许 direct_path 旧权重冒烟。
        'data_root': '<SIC_DATA_ROOT>',  # 包含 3DPW_MC/train 的官方数据根目录。
        'output_path': '<COMPLETION_RESULT_PATH>',  # 返回 SARS-Inter 的 V2 result pkl。
        'device': 'cuda:0',  # 可选 cuda:0 或 cpu。
        'mask_policy': 'allow_ood_explicit',  # strict_official_mc 或 allow_ood_explicit。
        'demonstration_seed': 42,  # 只从 3DPW_MC/train 稳定选择 demonstration。
        'demonstration_selection_policy': 'per_sample_fixed',  # 四个 F16 窗口共用同一 train demonstration。
        'coordinate_transform_mode': TRANSFORM_MODE,  # 正式比较使用项目 H36M17 到 SiC 的可逆坐标适配。
    }


def _hashable(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            '__ndarray__': {
                'dtype': str(array.dtype),
                'shape': list(array.shape),
                'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            }
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _hashable(child)
            for key, child in sorted(value.items(), key=lambda item: item[0])
            if key != 'input_binding_validated'
        }
    if isinstance(value, (list, tuple)):
        return [_hashable(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('result metadata must be finite')
    return value


def _package_hash(package, field_name):
    payload = {key: value for key, value in package.items() if key != field_name}
    encoded = json.dumps(
        _hashable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path):
    return _identity_sha256_file(path)


def _update_length_prefixed(digest, value):
    payload = bytes(value)
    digest.update(len(payload).to_bytes(8, byteorder='big', signed=False))
    digest.update(payload)


def _git_command(*arguments):
    """只为当前仓库进程声明 safe.directory，不修改全局 Git 配置。"""
    safe_root = PROJECT_ROOT.replace('\\', '/')
    return ['git', '-c', 'safe.directory={}'.format(safe_root)] + list(arguments)


def _validate_formal_coordinate_contract(contract):
    value = dict(contract or {})
    if value.get('skeleton') != 'H36M17':
        raise ValueError('formal coordinate adapter requires H36M17 query')
    if value.get('joint_order') != 'h36m17_sars_inter_project_order':
        raise ValueError(
            'formal coordinate adapter requires joint_order='
            'h36m17_sars_inter_project_order'
        )
    if value.get('axis_order') != [
        'x_lateral', 'y_depth', 'z_height'
    ]:
        raise ValueError(
            'formal coordinate adapter requires project Z-up axis_order'
        )


def _resolve_demonstration_selection_policy(transform_mode, configured):
    expected = (
        'per_sample_fixed' if transform_mode == TRANSFORM_MODE else 'per_window'
    )
    policy = str(configured or expected)
    if policy != expected:
        raise ValueError(
            '{} requires {} demonstration selection policy'.format(
                transform_mode, expected
            )
        )
    return policy


def _sample_input_selection_keys(query):
    masked = np.asarray(query['masked_keypoint'], dtype=np.float32)
    missing = np.asarray(query['missing_mask'], dtype=bool)
    keys = []
    for index in range(masked.shape[0]):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(masked[index]).tobytes(order='C'))
        digest.update(np.ascontiguousarray(missing[index]).tobytes(order='C'))
        keys.append(digest.hexdigest())
    return keys


def build_result_package(
    query,
    completed_keypoint,
    repository_commit,
    checkpoint_identity,
    runtime_metadata,
    method_metadata,
    provenance=None,
):
    completed = np.asarray(completed_keypoint, dtype=np.float32)
    if completed.shape != (len(query['sample_order']), 64, 17, 3):
        raise ValueError('completed_keypoint must match query sample order and F64 shape')
    if not np.isfinite(completed).all():
        raise ValueError('completed_keypoint must contain finite values')
    package = {
        'format': 'sars_inter_external_completion_result',
        'version': 2,
        'query_hash': query['query_hash'],
        'dataset_profile': query['dataset_profile'],
        'source_split': query['source_split'],
        'sample_order': list(query['sample_order']),
        'frame_count': 64,
        'joint_count': 17,
        'coordinate_contract': copy.deepcopy(query['coordinate_contract']),
        'completed_keypoint': completed,
        'method_name': 'skeleton_in_context',
        'repository_commit': str(repository_commit),
        'checkpoint_identity': str(checkpoint_identity),
        'runtime_metadata': copy.deepcopy(runtime_metadata),
        'method_metadata': copy.deepcopy(method_metadata),
        'provenance': copy.deepcopy(dict(provenance or {})),
    }
    package['result_hash'] = _package_hash(package, 'result_hash')
    return package


def _repository_identity():
    """尽力记录仓库状态；Git 元信息不可用时不阻断实验执行。"""
    try:
        commit = subprocess.check_output(
            _git_command('rev-parse', 'HEAD'), cwd=PROJECT_ROOT
        ).decode('ascii').strip()
        tracked_diff = subprocess.check_output(
            _git_command('diff', '--binary', 'HEAD', '--'), cwd=PROJECT_ROOT
        )
        untracked_output = subprocess.check_output(
            _git_command('ls-files', '--others', '--exclude-standard', '-z'),
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return 'unknown', None, None
    untracked_paths = [
        value.decode('utf-8', errors='surrogateescape')
        for value in untracked_output.split(b'\0')
        if value
    ]
    untracked_paths = [
        path for path in untracked_paths
        if '__pycache__' not in path.replace('\\', '/').split('/')
        and not path.endswith(('.pyc', '.pyo'))
    ]
    digest = hashlib.sha256()
    _update_length_prefixed(digest, commit.encode('ascii'))
    _update_length_prefixed(digest, b'tracked-diff')
    _update_length_prefixed(digest, tracked_diff)
    for relative_path in sorted(untracked_paths):
        absolute_path = os.path.join(PROJECT_ROOT, relative_path)
        if not os.path.isfile(absolute_path):
            continue
        file_digest = hashlib.sha256()
        file_size = 0
        with open(absolute_path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                file_size += len(chunk)
                file_digest.update(chunk)
        _update_length_prefixed(digest, b'untracked-file')
        _update_length_prefixed(
            digest,
            relative_path.encode('utf-8', errors='surrogateescape'),
        )
        _update_length_prefixed(
            digest,
            int(file_size).to_bytes(8, byteorder='big', signed=False),
        )
        _update_length_prefixed(digest, file_digest.digest())
    return commit, bool(tracked_diff or untracked_paths), digest.hexdigest()


def _load_model(
    source_config,
    data_root,
    checkpoint_path,
    device,
    checkpoint_identity_policy='require_mc_only',
):
    args = get_config(source_config)
    args.full_data.root_path = data_root
    args.use_partial_data = False
    args.data = copy.deepcopy(args.full_data)
    if args.backbone != 'SiC_dynamicTUP':
        raise ValueError('SARS adapter currently supports SiC_dynamicTUP checkpoint')
    model = load_backbone(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    training_identity = validate_completion_checkpoint_identity(
        checkpoint,
        source_config,
        policy=checkpoint_identity_policy,
    )
    state = checkpoint['model_pos']
    if state and all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(torch.device(device))
    model.sars_training_identity = training_identity
    return model


def _normalize_runtime_options(config):
    resolved = copy.deepcopy(dict(config))
    for name in ('data_root',):
        value = os.path.abspath(os.path.expanduser(str(resolved.get(name) or '')))
        if not value or '<' in value or '>' in value:
            raise ValueError('{} still contains a placeholder'.format(name))
        resolved[name] = value
    if not os.path.isdir(resolved['data_root']):
        raise FileNotFoundError(resolved['data_root'])
    resolved['device'] = str(resolved.get('device') or 'cuda:0')
    resolved['checkpoint_identity_policy'] = str(
        resolved.get('checkpoint_identity_policy') or 'require_mc_only'
    )
    if resolved['checkpoint_identity_policy'] not in {
        'require_mc_only', 'allow_legacy'
    }:
        raise ValueError('unsupported checkpoint identity policy')
    # 兼容旧配置字段，但 Git revision 只作为 provenance 记录，不再阻断运行。
    resolved['require_clean_repository'] = False
    transform_mode = resolved.get('coordinate_transform_mode')
    if transform_mode not in {'identity_h36m17', TRANSFORM_MODE}:
        raise ValueError('unsupported coordinate_transform_mode')
    resolved['demonstration_selection_policy'] = (
        _resolve_demonstration_selection_policy(
            transform_mode, resolved.get('demonstration_selection_policy')
        )
    )
    return resolved


def _resolve_runtime_paths(config):
    return _normalize_runtime_options(resolve_checkpoint_source(config))


def _reuse_checkpoint_source(config, cached):
    resolved = copy.deepcopy(dict(config))
    mode = resolved.get('checkpoint_source_mode')
    if mode is not None and str(mode) != cached['checkpoint_source_mode']:
        raise ValueError('shared completion checkpoint_source_mode mismatch')
    for name in (
        'checkpoint_identity_manifest_path', 'checkpoint_path', 'source_config'
    ):
        configured = str(resolved.get(name) or '').strip()
        if not configured or '<' in configured or '>' in configured:
            continue
        if os.path.abspath(os.path.expanduser(configured)) != cached.get(name):
            raise ValueError('shared completion {} mismatch'.format(name))
    expected_sha256 = str(resolved.get('checkpoint_sha256') or '').strip().lower()
    if expected_sha256 and expected_sha256 != cached['checkpoint_sha256']:
        raise ValueError('shared completion checkpoint_sha256 mismatch')
    resolved.update(copy.deepcopy(cached))
    return _normalize_runtime_options(resolved)


def _completion_policy(resolved):
    return {
        'inference_task': 'joint_completion',
        'window_policy': 'non_overlapping_f16',
        'mask_policy': str(
            resolved.get('mask_policy', 'allow_ood_explicit')
        ),
        'coordinate_transform_mode': str(
            resolved['coordinate_transform_mode']
        ),
        'demonstration_source_split': 'train',
        'demonstration_seed': int(resolved.get('demonstration_seed', 42)),
        'demonstration_selection_policy': str(
            resolved['demonstration_selection_policy']
        ),
        'observed_joint_policy': 'exact_restore',
        'checkpoint_identity_policy': str(
            resolved['checkpoint_identity_policy']
        ),
    }


def prepare_completion_runtime(config, runtime=None, load_model=True):
    """准备可跨 query 复用、但不保存 query 私有状态的运行时。"""
    shared = runtime if runtime is not None else {}
    cached_source = shared.get('resolved_checkpoint_source')
    if cached_source is None:
        resolved = _resolve_runtime_paths(config)
        shared['resolved_checkpoint_source'] = {
            name: copy.deepcopy(resolved.get(name))
            for name in (
                'checkpoint_source_mode',
                'checkpoint_identity_manifest_path',
                'checkpoint_identity_manifest_sha256',
                'checkpoint_path',
                'source_config',
                'checkpoint_sha256',
                'checkpoint_identity_policy',
                'checkpoint_training_identity',
            )
        }
    else:
        resolved = _reuse_checkpoint_source(config, cached_source)
    signature = (
        resolved['source_config'], resolved['data_root'],
        resolved['checkpoint_path'], resolved['device'],
        resolved.get('checkpoint_identity_manifest_sha256'),
        resolved['coordinate_transform_mode'],
        resolved['demonstration_selection_policy'],
        int(resolved.get('demonstration_seed', 42)),
        str(resolved.get('mask_policy', 'allow_ood_explicit')),
        resolved['checkpoint_identity_policy'],
    )
    previous = shared.get('runtime_signature')
    if previous is not None and tuple(previous) != signature:
        raise ValueError('shared completion runtime configuration mismatch')
    shared['runtime_signature'] = signature
    if 'prompt_pool' not in shared:
        shared['prompt_pool'] = build_train_prompt_pool(
            resolved['data_root'], resolved['source_config']
        )
    if 'repository_identity' not in shared:
        shared['repository_identity'] = _repository_identity()
    if 'checkpoint_sha256' not in shared:
        shared['checkpoint_sha256'] = resolved['checkpoint_sha256']
    if (
        not load_model
        and resolved['checkpoint_identity_policy'] == 'require_mc_only'
        and 'checkpoint_training_identity' not in shared
    ):
        checkpoint = torch.load(resolved['checkpoint_path'], map_location='cpu')
        shared['checkpoint_training_identity'] = (
            validate_completion_checkpoint_identity(
                checkpoint,
                resolved['source_config'],
                policy=resolved['checkpoint_identity_policy'],
            )
        )
    if 'provenance' not in shared:
        commit, dirty, worktree_hash = shared['repository_identity']
        prompt_pool = shared['prompt_pool']
        shared['provenance'] = {
            'official_upstream': {
                'repository_url': OFFICIAL_UPSTREAM_REPOSITORY_URL,
                'commit': OFFICIAL_UPSTREAM_COMMIT,
            },
            'adapter_fork': {
                'repository_url': ADAPTER_FORK_REPOSITORY_URL,
                'commit': commit,
                'repository_dirty': dirty,
                'repository_worktree_sha256': worktree_hash,
            },
            'checkpoint': {
                'sha256': shared['checkpoint_sha256'],
                'source_mode': resolved['checkpoint_source_mode'],
                'identity_manifest_sha256': resolved.get(
                    'checkpoint_identity_manifest_sha256'
                ),
                'training_identity': copy.deepcopy(
                    shared.get('checkpoint_training_identity')
                ),
            },
            'prompt_pool': {
                'manifest_sha256': prompt_pool[
                    'prompt_pool_manifest_sha256'
                ],
                'file_count': int(prompt_pool['prompt_pool_file_count']),
                'source_config_sha256': prompt_pool[
                    'source_config_sha256'
                ],
            },
            'completion_policy': _completion_policy(resolved),
        }
    if load_model and 'model' not in shared:
        shared['model'] = _load_model(
            resolved['source_config'], resolved['data_root'],
            resolved['checkpoint_path'], resolved['device'],
            checkpoint_identity_policy=resolved[
                'checkpoint_identity_policy'
            ],
        )
        shared['checkpoint_training_identity'] = copy.deepcopy(
            getattr(shared['model'], 'sars_training_identity', None)
        )
        shared['provenance']['checkpoint']['training_identity'] = copy.deepcopy(
            shared['checkpoint_training_identity']
        )
        if _sha256_file(resolved['checkpoint_path']) != shared['checkpoint_sha256']:
            raise RuntimeError('SiC checkpoint changed while loading the model')
        if (
            _sha256_file(resolved['source_config'])
            != shared['prompt_pool']['source_config_sha256']
        ):
            raise RuntimeError('SiC source config changed while loading the model')
    return shared


def verify_completion_runtime_assets(config, runtime):
    """在 series 结束前复核冻结资产，防止长任务期间身份漂移。"""
    resolved = _resolve_runtime_paths(config)
    shared = dict(runtime or {})
    cached_manifest_hash = (
        (shared.get('provenance') or {}).get('checkpoint') or {}
    ).get('identity_manifest_sha256')
    if (
        resolved.get('checkpoint_identity_manifest_sha256')
        != cached_manifest_hash
    ):
        raise RuntimeError(
            'SiC checkpoint identity manifest changed during completion series'
        )
    if _sha256_file(resolved['checkpoint_path']) != shared.get(
        'checkpoint_sha256'
    ):
        raise RuntimeError('SiC checkpoint changed during completion series')
    current_pool = build_train_prompt_pool(
        resolved['data_root'], resolved['source_config']
    )
    frozen_pool = shared.get('prompt_pool') or {}
    for field in (
        'prompt_pool_manifest_sha256',
        'prompt_pool_file_count',
        'source_config_sha256',
    ):
        if current_pool.get(field) != frozen_pool.get(field):
            raise RuntimeError(
                'SiC prompt pool changed during completion series: {}'.format(
                    field
                )
            )
    return True


def run_completion(config, runtime=None):
    resolved = copy.deepcopy(dict(config))
    for name in ('query_path', 'data_root', 'output_path'):
        value = os.path.abspath(os.path.expanduser(str(resolved.get(name) or '')))
        if not value or '<' in value or '>' in value:
            raise ValueError('{} still contains a placeholder'.format(name))
        resolved[name] = value
    for name in ('query_path',):
        if not os.path.isfile(resolved[name]):
            raise FileNotFoundError(resolved[name])
    if not os.path.isdir(resolved['data_root']):
        raise FileNotFoundError(resolved['data_root'])
    query = load_completion_query(resolved['query_path'])
    transform_mode = resolved.get('coordinate_transform_mode')
    if transform_mode not in {'identity_h36m17', TRANSFORM_MODE}:
        raise ValueError('unsupported coordinate_transform_mode')
    if transform_mode == TRANSFORM_MODE:
        _validate_formal_coordinate_contract(query.get('coordinate_contract'))
    selection_policy = _resolve_demonstration_selection_policy(
        transform_mode, resolved.get('demonstration_selection_policy')
    )
    preview = {
        'status': 'planned' if resolved.get('dry_run', True) else 'running',
        'query_hash': query['query_hash'],
        'sample_count': len(query['sample_order']),
        'output_path': resolved['output_path'],
    }
    if resolved.get('dry_run', True):
        checkpoint_source = _resolve_runtime_paths(resolved)
        checkpoint = torch.load(
            checkpoint_source['checkpoint_path'], map_location='cpu'
        )
        training_identity = validate_completion_checkpoint_identity(
            checkpoint,
            checkpoint_source['source_config'],
            policy=checkpoint_source['checkpoint_identity_policy'],
        )
        preview.update({
            'checkpoint_source_mode': checkpoint_source[
                'checkpoint_source_mode'
            ],
            'checkpoint_sha256': checkpoint_source['checkpoint_sha256'],
            'checkpoint_identity_manifest_sha256': checkpoint_source.get(
                'checkpoint_identity_manifest_sha256'
            ),
            'checkpoint_training_identity': copy.deepcopy(training_identity),
        })
        return preview
    resolved['demonstration_selection_policy'] = selection_policy
    shared = prepare_completion_runtime(resolved, runtime=runtime, load_model=True)
    commit, dirty, worktree_hash = shared['repository_identity']
    model = shared['model']
    prompt_provider = build_train_prompt_provider(
        resolved['data_root'], resolved['source_config'],
        seed=int(resolved.get('demonstration_seed', 42)),
        selection_policy=selection_policy,
        selection_keys=_sample_input_selection_keys(query),
        prompt_pool=shared['prompt_pool'],
    )
    started = time.perf_counter()
    completed, inference_metadata = complete_f64_with_model(
        model=model,
        masked_keypoint=query['masked_keypoint'],
        missing_mask=query['missing_mask'],
        prompt_provider=prompt_provider,
        device=resolved['device'],
        mask_policy=resolved.get('mask_policy', 'allow_ood_explicit'),
        coordinate_transform_mode=transform_mode,
    )
    elapsed = time.perf_counter() - started
    final_identity = _repository_identity()
    checkpoint_hash = shared['checkpoint_sha256']
    runtime = {
        'total_seconds': float(elapsed),
        'device': resolved['device'],
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
    }
    method_metadata = {
        'inference_task': 'joint_completion',
        'demonstration_source_splits': ['train'],
        'demonstration_seed': int(resolved.get('demonstration_seed', 42)),
        'demonstration_selection_policy': selection_policy,
        'window_policy': inference_metadata['window_policy'],
        'window_records': inference_metadata['windows'],
        'mask_policy': inference_metadata['mask_policy'],
        'coordinate_transform_mode': resolved['coordinate_transform_mode'],
        'sample_transforms': inference_metadata['sample_transforms'],
        'boundary_diagnostics': inference_metadata['boundary_diagnostics'],
        'observed_restoration_max_abs_error': inference_metadata[
            'observed_restoration_max_abs_error'
        ],
        'repository_dirty': dirty,
        'repository_worktree_sha256': worktree_hash,
        'repository_changed_during_execution': (
            final_identity != (commit, dirty, worktree_hash)
        ),
    }
    package = build_result_package(
        query=query,
        completed_keypoint=completed,
        repository_commit=commit,
        checkpoint_identity=checkpoint_hash,
        runtime_metadata=runtime,
        method_metadata=method_metadata,
        provenance=shared['provenance'],
    )
    output_dir = os.path.dirname(resolved['output_path'])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(resolved['output_path'], 'wb') as stream:
        pickle.dump(package, stream, protocol=EXTERNAL_PICKLE_PROTOCOL)
    preview.update({
        'status': 'completed',
        'result_hash': package['result_hash'],
        'checkpoint_sha256': checkpoint_hash,
        'runtime_seconds': float(elapsed),
    })
    return preview


def main():
    summary = run_completion(build_direct_run_config())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    main()
