"""使用显式 missing_mask 执行 F64 Skeleton-in-Context 补全。"""

from __future__ import annotations

import hashlib
import os
import numpy as np
import torch

from lib.data.dataset import skel_to_h36m
from lib.utils.tools import get_config, read_pkl
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


def build_train_prompt_provider(data_root, source_config, seed=42):
    """从官方 3DPW_MC/train 稳定选择 demonstration。"""
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
    joint_map = args.amass_to_h36m

    def provider(sample_index, window_index, window_mask):
        token = '{}:{}:{}'.format(int(seed), sample_index, window_index)
        index = int.from_bytes(
            hashlib.sha256(token.encode('utf-8')).digest()[:8], 'big'
        ) % len(files)
        path = files[index]
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
            'sha256': _sha256_file(path),
            'selection_seed': int(seed),
        }
        return (
            np.asarray(prompt_input, dtype=np.float32),
            np.asarray(prompt_target, dtype=np.float32),
            metadata,
        )

    return provider


def _prompt_arrays(prompt_provider, sample_index, window_index, window_mask):
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
    masked_prompt = np.where(window_mask[..., None], 0.0, prompt_input)
    return masked_prompt, prompt_target, metadata


def complete_f64_with_model(
    model,
    masked_keypoint,
    missing_mask,
    prompt_provider,
    device='cuda:0',
    mask_policy='allow_ood_explicit',
):
    """按四个非重叠 F16 窗口补全，并精确恢复全部可见关节。"""
    masked = np.asarray(masked_keypoint, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=bool)
    if masked.ndim != 4 or masked.shape[1:] != (64, 17, 3):
        raise ValueError('masked_keypoint must be (samples,64,17,3)')
    if missing.shape != masked.shape[:-1]:
        raise ValueError('missing_mask must be (samples,64,17)')
    if mask_policy not in {'strict_official_mc', 'allow_ood_explicit'}:
        raise ValueError('unsupported mask_policy')
    if not np.isfinite(masked).all():
        raise ValueError('masked_keypoint must contain finite values')

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    completed = np.array(masked, copy=True)
    window_records = []
    demonstration_splits = set()
    for sample_index in range(masked.shape[0]):
        compatibility = analyze_mask_compatibility(missing[sample_index])
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
            window_mask = missing[sample_index, start:end]
            prompt_input, prompt_target, prompt_metadata = _prompt_arrays(
                prompt_provider, sample_index, window_index, window_mask
            )
            demonstration_splits.add(prompt_metadata['source_split'])
            query_input = masked[sample_index, start:end]
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
            completed[sample_index, start:end] = restore_observed_joints(
                query_input, generated, window_mask
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
        'missing_mask_semantics': 'True=missing',
        'demonstration_source_splits': sorted(demonstration_splits),
        'observed_restoration_max_abs_error': observed_error,
    }
    return completed, metadata


__all__ = ['build_train_prompt_provider', 'complete_f64_with_model']
