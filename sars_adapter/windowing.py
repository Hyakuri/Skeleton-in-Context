"""SARS F64 与 SiC F16 之间的确定性窗口和 mask 处理。"""

from __future__ import annotations

import numpy as np


def f64_windows(frame_count=64, window_size=16):
    if int(frame_count) != 64 or int(window_size) != 16:
        raise ValueError('current SiC adapter requires F64 split into F16 windows')
    return [(start, start + window_size) for start in range(0, frame_count, window_size)]


def analyze_mask_compatibility(missing_mask):
    mask = np.asarray(missing_mask, dtype=bool)
    if mask.shape != (64, 17):
        raise ValueError('missing_mask must be (64,17)')
    reasons = []
    missing_counts = []
    for start, end in f64_windows():
        window = mask[start:end]
        is_constant = bool(np.all(window == window[0:1]))
        if not is_constant:
            reasons.append('time_varying_within_window')
        missing_count = int(window[0].sum()) if is_constant else int(
            np.any(window, axis=0).sum()
        )
        missing_counts.append(missing_count)
        if missing_count not in {6, 10}:
            reasons.append('unsupported_missing_count')
    if np.any(mask[:, [0, 16]]):
        reasons.append('unsupported_boundary_joint')
    reasons = list(dict.fromkeys(reasons))
    return {
        'official_mc_compatible': not reasons,
        'reasons': reasons,
        'missing_count': int(mask.sum()),
        'window_missing_joint_counts': missing_counts,
        'missing_joint_ids': np.where(np.any(mask, axis=0))[0].astype(int).tolist(),
    }


def restore_observed_joints(masked_keypoint, generated_keypoint, missing_mask):
    masked = np.asarray(masked_keypoint, dtype=np.float32)
    generated = np.asarray(generated_keypoint, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=bool)
    if masked.shape != generated.shape or masked.shape != missing.shape + (3,):
        raise ValueError('masked/generated/missing shapes are incompatible')
    return np.where(missing[..., None], generated, masked)


__all__ = ['analyze_mask_compatibility', 'f64_windows', 'restore_observed_joints']
