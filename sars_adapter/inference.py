"""使用显式 missing_mask 执行 F64 Skeleton-in-Context 补全。"""

from __future__ import annotations

import hashlib
import os
import numpy as np
import torch

from lib.data.dataset import skel_to_h36m
from lib.utils.tools import get_config, read_pkl
from sars_adapter.coordinate_adapter import (
    TRANSFORM_MODE,
    inverse_project_h36m17_window,
    prepare_project_h36m17_sample,
    project_to_sic_joints,
)
from sars_adapter.windowing import (
    analyze_mask_compatibility,
    f64_windows,
    restore_observed_joints,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_train_prompt_pool(data_root, source_config):
    """冻结 3DPW_MC/train prompt pool 的内容身份与关节映射。"""
    train_dir = os.path.join(os.path.abspath(data_root), '3DPW_MC', 'train')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(train_dir)
    files = sorted(
        os.path.join(train_dir, name)
        for name in os.listdir(train_dir)
        if os.path.isfile(os.path.join(train_dir, name))
    )
    if not files:
        raise ValueError('3DPW_MC/train contains no demonstration files')
    args = get_config(source_config)
    pool_digest = hashlib.sha256()
    file_hashes = {}
    for path in files:
        file_hash = _sha256_file(path)
        file_hashes[path] = file_hash
        record = '{}\0{}\0{}'.format(
            os.path.basename(path), os.path.getsize(path), file_hash
        ).encode('utf-8')
        pool_digest.update(len(record).to_bytes(8, 'big'))
        pool_digest.update(record)
    return {
        'files': files,
        'file_sha256_by_path': file_hashes,
        'joint_map': args.amass_to_h36m,
        'prompt_pool_file_count': int(len(files)),
        'prompt_pool_manifest_sha256': pool_digest.hexdigest(),
        'source_config_sha256': _sha256_file(source_config),
    }


def build_train_prompt_provider(
    data_root, source_config, seed=42, selection_policy='per_window',
    selection_keys=None, prompt_pool=None,
):
    """从官方 3DPW_MC/train 稳定选择 demonstration。"""
    pool = dict(prompt_pool or build_train_prompt_pool(data_root, source_config))
    files = list(pool['files'])
    joint_map = pool['joint_map']
    if selection_policy not in {'per_window', 'per_sample_fixed'}:
        raise ValueError('unsupported demonstration selection_policy')
    if selection_keys is not None:
        selection_keys = [str(value) for value in selection_keys]
    pool_hash = str(pool['prompt_pool_manifest_sha256'])
    source_config_hash = str(pool['source_config_sha256'])
    file_hashes = dict(pool.get('file_sha256_by_path') or {})

    def provider(sample_index, window_index, window_mask):
        if selection_policy == 'per_sample_fixed':
            if selection_keys is not None:
                if sample_index >= len(selection_keys):
                    raise IndexError('demonstration selection key is missing')
                selection_key = selection_keys[sample_index]
            else:
                selection_key = str(sample_index)
            token = '{}:{}'.format(int(seed), selection_key)
        else:
            token = '{}:{}:{}'.format(int(seed), sample_index, window_index)
        index = int.from_bytes(
            hashlib.sha256(token.encode('utf-8')).digest()[:8], 'big'
        ) % len(files)
        path = files[index]
        expected_file_hash = file_hashes.get(path)
        actual_file_hash = _sha256_file(path)
        if not expected_file_hash or actual_file_hash != expected_file_hash:
            raise RuntimeError(
                'prompt demonstration changed after pool freeze: {}'.format(path)
            )
        payload = read_pkl(path)
        prompt_input = skel_to_h36m(
            np.asarray(payload['data_input']), joint_map
        )
        prompt_target = skel_to_h36m(
            np.asarray(payload['data_label']), joint_map
        )
        if prompt_input.shape != (16, 17, 3) or prompt_target.shape != (16, 17, 3):
            raise ValueError(
                '3DPW_MC demonstration must resolve to (16,17,3): {}'.format(path)
            )
        metadata = {
            'source_split': 'train',
            'identity': os.path.basename(path),
            'sha256': actual_file_hash,
            'selection_seed': int(seed),
            'selection_policy': selection_policy,
            'selection_key_sha256': hashlib.sha256(
                token.encode('utf-8')
            ).hexdigest(),
            'prompt_pool_file_count': int(len(files)),
            'prompt_pool_manifest_sha256': pool_hash,
            'source_config_sha256': source_config_hash,
        }
        return (
            np.asarray(prompt_input, dtype=np.float32),
            np.asarray(prompt_target, dtype=np.float32),
            metadata,
        )

    provider.prompt_pool_file_count = int(len(files))
    provider.prompt_pool_manifest_sha256 = pool_hash
    provider.source_config_sha256 = source_config_hash
    return provider


def _raw_prompt_arrays(prompt_provider, sample_index, window_index, window_mask):
    prompt_input, prompt_target, metadata = prompt_provider(
        sample_index, window_index, window_mask
    )
    prompt_input = np.asarray(prompt_input, dtype=np.float32)
    prompt_target = np.asarray(prompt_target, dtype=np.float32)
    if prompt_input.shape != (16, 17, 3) or prompt_target.shape != (16, 17, 3):
        raise ValueError('SiC demonstration input/target must be (16,17,3)')
    metadata = dict(metadata or {})
    if metadata.get('source_split') != 'train':
        raise ValueError('SiC demonstrations must come from train split')
    return prompt_input, prompt_target, metadata


def _prompt_arrays(prompt_provider, sample_index, window_index, window_mask):
    prompt_input, prompt_target, metadata = _raw_prompt_arrays(
        prompt_provider, sample_index, window_index, window_mask
    )
    masked_prompt = np.where(window_mask[..., None], 0.0, prompt_input)
    return masked_prompt, prompt_target, metadata


def _boundary_diagnostics(completed, missing_mask):
    records = []
    for sample_index in range(completed.shape[0]):
        sample = completed[sample_index]
        for boundary in (16, 32, 48):
            position_jump = np.linalg.norm(
                sample[boundary] - sample[boundary - 1], axis=-1
            )
            velocity_before = sample[boundary - 1] - sample[boundary - 2]
            velocity_after = sample[boundary + 1] - sample[boundary]
            velocity_jump = np.linalg.norm(
                velocity_after - velocity_before, axis=-1
            )
            missing_joints = np.any(
                missing_mask[
                    sample_index, boundary - 2:boundary + 2
                ],
                axis=0,
            )
            missing_position = position_jump[missing_joints]
            missing_velocity = velocity_jump[missing_joints]
            record = {
                'sample_index': int(sample_index),
                'boundary_frame': int(boundary),
                'position_jump_mean': float(np.mean(position_jump)),
                'position_jump_max': float(np.max(position_jump)),
                'velocity_jump_mean': float(np.mean(velocity_jump)),
                'velocity_jump_max': float(np.max(velocity_jump)),
                'missing_joint_count': int(np.sum(missing_joints)),
                'missing_position_jump_mean': (
                    float(np.mean(missing_position))
                    if missing_position.size else 0.0
                ),
                'missing_position_jump_max': (
                    float(np.max(missing_position))
                    if missing_position.size else 0.0
                ),
                'missing_velocity_jump_mean': (
                    float(np.mean(missing_velocity))
                    if missing_velocity.size else 0.0
                ),
                'missing_velocity_jump_max': (
                    float(np.max(missing_velocity))
                    if missing_velocity.size else 0.0
                ),
            }
            records.append(record)
    return records


def complete_f64_with_model(
    model,
    masked_keypoint,
    missing_mask,
    prompt_provider,
    device='cuda:0',
    mask_policy='allow_ood_explicit',
    coordinate_transform_mode='identity_h36m17',
):
    """按四个非重叠 F16 窗口补全，并精确恢复全部可见关节。"""
    masked = np.asarray(masked_keypoint, dtype=np.float32)
    raw_missing = np.asarray(missing_mask)
    if raw_missing.dtype != np.bool_:
        raise ValueError('missing_mask dtype must be bool')
    missing = np.asarray(raw_missing, dtype=bool)
    if masked.ndim != 4 or masked.shape[1:] != (64, 17, 3):
        raise ValueError('masked_keypoint must be (samples,64,17,3)')
    if missing.shape != masked.shape[:-1]:
        raise ValueError('missing_mask must be (samples,64,17)')
    if mask_policy not in {'strict_official_mc', 'allow_ood_explicit'}:
        raise ValueError('unsupported mask_policy')
    if coordinate_transform_mode not in {'identity_h36m17', TRANSFORM_MODE}:
        raise ValueError('unsupported coordinate_transform_mode')
    if not np.isfinite(masked).all():
        raise ValueError('masked_keypoint must contain finite values')

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    completed = np.array(masked, copy=True)
    window_records = []
    sample_transforms = []
    demonstration_splits = set()
    for sample_index in range(masked.shape[0]):
        formal_mode = coordinate_transform_mode == TRANSFORM_MODE
        if formal_mode:
            first_model_mask = np.asarray(
                project_to_sic_joints(missing[sample_index, :16]), dtype=bool
            )
            prompt_input_raw, prompt_target_raw, prompt_metadata = (
                _raw_prompt_arrays(
                    prompt_provider, sample_index, 0, first_model_mask
                )
            )
            query_f64, model_missing, transform_state, transform_metadata = (
                prepare_project_h36m17_sample(
                    masked[sample_index], missing[sample_index], prompt_input_raw
                )
            )
            sample_transforms.append(transform_metadata)
        else:
            query_f64 = masked[sample_index]
            model_missing = missing[sample_index]
            transform_state = None
            prompt_input_raw = None
            prompt_target_raw = None
            prompt_metadata = None
        compatibility = analyze_mask_compatibility(model_missing)
        if (
            mask_policy == 'strict_official_mc'
            and not compatibility['official_mc_compatible']
        ):
            raise ValueError(
                'missing_mask is outside official MC training support: {}'.format(
                    compatibility['reasons']
                )
            )
        for window_index, (start, end) in enumerate(f64_windows()):
            window_mask = model_missing[start:end]
            if formal_mode:
                prompt_input = np.where(
                    window_mask[..., None], 0.0, prompt_input_raw
                )
                prompt_target = prompt_target_raw
            else:
                prompt_input, prompt_target, prompt_metadata = _prompt_arrays(
                    prompt_provider, sample_index, window_index, window_mask
                )
            demonstration_splits.add(prompt_metadata['source_split'])
            query_input = query_f64[start:end]
            prompt = np.concatenate([prompt_input, prompt_target], axis=0)
            query = np.concatenate(
                [query_input, np.zeros_like(query_input)], axis=0
            )
            prompt_tensor = torch.from_numpy(prompt[None]).to(torch_device)
            query_tensor = torch.from_numpy(query[None]).to(torch_device)
            with torch.no_grad():
                prediction, _ = model(prompt_tensor, query_tensor, None)
            generated = prediction.detach().cpu().numpy()[0]
            if generated.shape != (16, 17, 3) or not np.isfinite(generated).all():
                raise ValueError('SiC prediction must be finite (16,17,3)')
            if formal_mode:
                generated = inverse_project_h36m17_window(
                    generated, transform_state, start, end
                )
                restore_input = masked[sample_index, start:end]
                restore_mask = missing[sample_index, start:end]
            else:
                restore_input = query_input
                restore_mask = window_mask
            completed[sample_index, start:end] = restore_observed_joints(
                restore_input, generated, restore_mask
            )
            window_records.append({
                'sample_index': sample_index,
                'window_index': window_index,
                'start': start,
                'end': end,
                'mask_compatibility': compatibility,
                'demonstration': prompt_metadata,
            })
    observed_error = float(
        np.max(np.abs(completed[~missing] - masked[~missing]))
    ) if np.any(~missing) else 0.0
    metadata = {
        'window_policy': 'non_overlapping_f16',
        'windows': window_records,
        'mask_policy': mask_policy,
        'coordinate_transform_mode': coordinate_transform_mode,
        'sample_transforms': sample_transforms,
        'boundary_diagnostics': _boundary_diagnostics(completed, missing),
        'missing_mask_semantics': 'True=missing',
        'demonstration_source_splits': sorted(demonstration_splits),
        'observed_restoration_max_abs_error': observed_error,
    }
    return completed, metadata


__all__ = [
    'build_train_prompt_pool', 'build_train_prompt_provider',
    'complete_f64_with_model',
]
