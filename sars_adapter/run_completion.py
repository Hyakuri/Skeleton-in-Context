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
from sars_adapter.contracts import load_completion_query
from sars_adapter.coordinate_adapter import TRANSFORM_MODE
from sars_adapter.inference import (
    build_train_prompt_provider,
    complete_f64_with_model,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_PICKLE_PROTOCOL = 4


def build_direct_run_config():
    """集中放置外部补全推理需要手动调整的参数。"""
    return {
        'dry_run': True,  # True=仅检查输入；False=执行真实 GPU 推理。
        'query_path': '<COMPLETION_QUERY_PATH>',  # SARS-Inter V2 query，不得传评价 sidecar。
        'checkpoint_path': '<SIC_CHECKPOINT_PATH>',  # SiC latest/best checkpoint。
        'source_config': os.path.join(PROJECT_ROOT, 'configs', 'default.yaml'),  # 训练使用的官方基础配置。
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
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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
    }
    package['result_hash'] = _package_hash(package, 'result_hash')
    return package


def _repository_identity():
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


def _load_model(source_config, data_root, checkpoint_path, device):
    args = get_config(source_config)
    args.full_data.root_path = data_root
    args.use_partial_data = False
    args.data = copy.deepcopy(args.full_data)
    if args.backbone != 'SiC_dynamicTUP':
        raise ValueError('SARS adapter currently supports SiC_dynamicTUP checkpoint')
    model = load_backbone(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['model_pos']
    if state and all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(torch.device(device))


def run_completion(config):
    resolved = copy.deepcopy(dict(config))
    for name in ('query_path', 'checkpoint_path', 'source_config', 'data_root', 'output_path'):
        value = os.path.abspath(os.path.expanduser(str(resolved.get(name) or '')))
        if not value or '<' in value or '>' in value:
            raise ValueError('{} still contains a placeholder'.format(name))
        resolved[name] = value
    for name in ('query_path', 'checkpoint_path', 'source_config'):
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
        return preview
    commit, dirty, worktree_hash = _repository_identity()
    model = _load_model(
        resolved['source_config'], resolved['data_root'],
        resolved['checkpoint_path'], resolved['device']
    )
    prompt_provider = build_train_prompt_provider(
        resolved['data_root'], resolved['source_config'],
        seed=int(resolved.get('demonstration_seed', 42)),
        selection_policy=selection_policy,
        selection_keys=_sample_input_selection_keys(query),
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
    if final_identity != (commit, dirty, worktree_hash):
        raise RuntimeError(
            'repository worktree changed during SiC inference; result was not saved'
        )
    checkpoint_hash = _sha256_file(resolved['checkpoint_path'])
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
    }
    package = build_result_package(
        query=query,
        completed_keypoint=completed,
        repository_commit=commit,
        checkpoint_identity=checkpoint_hash,
        runtime_metadata=runtime,
        method_metadata=method_metadata,
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
