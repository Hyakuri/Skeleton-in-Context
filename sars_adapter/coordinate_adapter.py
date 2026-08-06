"""SARS 项目 H36M17 与 SiC 标准 H36M17 之间的可逆坐标适配。"""

from __future__ import annotations

import hashlib

import numpy as np


PROJECT_TO_SIC_JOINTS = (
    0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13,
)
SIC_TO_PROJECT_JOINTS = PROJECT_TO_SIC_JOINTS
SIC_EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)
AXIS_MATRIX_PROJECT_TO_SIC = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)
TRANSFORM_MODE = 'project_h36m17_prompt_aligned_v1'


def _joint_axis(array):
    value = np.asarray(array)
    if value.ndim >= 2 and value.shape[-2] == 17:
        return -2
    if value.ndim >= 1 and value.shape[-1] == 17:
        return -1
    raise ValueError('joint array must contain a H36M17 axis')


def project_to_sic_joints(array):
    """把项目左侧优先顺序转换为 SiC/MotionBERT 标准顺序。"""
    value = np.asarray(array)
    return np.take(value, PROJECT_TO_SIC_JOINTS, axis=_joint_axis(value))


def sic_to_project_joints(array):
    """把 SiC/MotionBERT 标准顺序还原为项目 H36M17 顺序。"""
    value = np.asarray(array)
    return np.take(value, SIC_TO_PROJECT_JOINTS, axis=_joint_axis(value))


def project_axes_to_sic(array):
    """执行右手系旋转：(x, y_depth, z_height) -> (x, z, -y)。"""
    value = np.asarray(array)
    if value.shape[-1] != 3:
        raise ValueError('coordinate array must end with XYZ')
    return np.stack((value[..., 0], value[..., 2], -value[..., 1]), axis=-1)


def sic_axes_to_project(array):
    """执行逆旋转：(x, y_height, z_depth) -> (x, -z, y)。"""
    value = np.asarray(array)
    if value.shape[-1] != 3:
        raise ValueError('coordinate array must end with XYZ')
    return np.stack((value[..., 0], -value[..., 2], value[..., 1]), axis=-1)


def _array_sha256(array):
    value = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def _interpolate_root_trajectory(keypoint, missing_mask):
    frame_count = keypoint.shape[0]
    root = np.zeros((frame_count, 3), dtype=np.float32)
    available = np.zeros(frame_count, dtype=bool)
    source = np.full(frame_count, 'missing', dtype=object)

    pelvis_visible = ~missing_mask[:, 0]
    root[pelvis_visible] = keypoint[pelvis_visible, 0]
    available[pelvis_visible] = True
    source[pelvis_visible] = 'pelvis'

    hip_visible = missing_mask[:, 0] & ~missing_mask[:, 1] & ~missing_mask[:, 4]
    root[hip_visible] = (
        keypoint[hip_visible, 1] + keypoint[hip_visible, 4]
    ) * 0.5
    available[hip_visible] = True
    source[hip_visible] = 'hip_midpoint'

    known = np.flatnonzero(available)
    if known.size == 0:
        raise ValueError('sample has no visible root anchor')
    frame_axis = np.arange(frame_count, dtype=np.float64)
    for coordinate in range(3):
        root[:, coordinate] = np.interp(
            frame_axis, known.astype(np.float64), root[known, coordinate]
        ).astype(np.float32)
    source[~available] = 'temporal_interpolation'
    return root, {
        'pelvis_frame_count': int(np.sum(source == 'pelvis')),
        'hip_midpoint_frame_count': int(np.sum(source == 'hip_midpoint')),
        'interpolated_frame_count': int(np.sum(source == 'temporal_interpolation')),
    }


def _visible_bone_scale(keypoint, missing_mask, field_name):
    values = []
    visible_edge_count = 0
    for parent, child in SIC_EDGES:
        visible = ~(missing_mask[:, parent] | missing_mask[:, child])
        if not np.any(visible):
            continue
        lengths = np.linalg.norm(
            keypoint[visible, child] - keypoint[visible, parent], axis=-1
        )
        valid_lengths = lengths[lengths > 1e-8]
        if valid_lengths.size:
            visible_edge_count += 1
            values.extend(valid_lengths.tolist())
    if visible_edge_count < 3:
        raise ValueError(
            '{} has insufficient visible bone topology: {} edges'.format(
                field_name, visible_edge_count
            )
        )
    if len(values) < 16:
        raise ValueError(
            '{} has insufficient visible bone support: {}'.format(
                field_name, len(values)
            )
        )
    scale = float(np.median(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError('{} visible bone scale is invalid'.format(field_name))
    return scale, len(values), visible_edge_count


def prepare_project_h36m17_sample(masked_keypoint, missing_mask, prompt_input):
    """在整条 F64 上构建四个窗口共享的可逆 SiC 坐标框架。"""
    masked = np.asarray(masked_keypoint, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=bool)
    prompt = np.asarray(prompt_input, dtype=np.float32)
    if masked.shape != (64, 17, 3) or missing.shape != (64, 17):
        raise ValueError('formal coordinate adapter requires F64 H36M17 input')
    if prompt.shape != (16, 17, 3):
        raise ValueError('SiC demonstration input must be (16,17,3)')
    if not np.isfinite(masked).all() or not np.isfinite(prompt).all():
        raise ValueError('coordinate adapter inputs must contain finite values')

    sic_missing = np.asarray(project_to_sic_joints(missing), dtype=bool)
    sic_masked = np.asarray(
        project_axes_to_sic(project_to_sic_joints(masked)), dtype=np.float32
    )
    query_root, root_counts = _interpolate_root_trajectory(
        sic_masked, sic_missing
    )
    prompt_f64 = np.tile(prompt, (4, 1, 1))
    prompt_root = np.asarray(prompt_f64[:, 0], dtype=np.float32)
    sample_scale, sample_value_count, sample_edge_count = _visible_bone_scale(
        sic_masked, sic_missing, 'sample'
    )
    reference_scale, reference_value_count, reference_edge_count = (
        _visible_bone_scale(
            prompt_f64, sic_missing, 'demonstration'
        )
    )
    scale_ratio = float(reference_scale / sample_scale)

    canonical = (
        (sic_masked - query_root[:, None, :]) * scale_ratio
        + prompt_root[:, None, :]
    ).astype(np.float32)
    canonical[sic_missing] = 0.0
    state = {
        'transform_mode': TRANSFORM_MODE,
        'query_root': query_root,
        'prompt_root': prompt_root,
        'scale_ratio': scale_ratio,
    }
    metadata = {
        'transform_mode': TRANSFORM_MODE,
        'transform_version': 1,
        'project_to_sic_joint_permutation': list(PROJECT_TO_SIC_JOINTS),
        'axis_matrix_project_to_sic': AXIS_MATRIX_PROJECT_TO_SIC.tolist(),
        'root_policy': 'pelvis_then_hip_midpoint_temporal_interpolation',
        'root_anchor_counts': root_counts,
        'query_root_sha256': _array_sha256(query_root),
        'prompt_root_sha256': _array_sha256(prompt_root),
        'sample_scale': sample_scale,
        'reference_scale': reference_scale,
        'scale_ratio': scale_ratio,
        'sample_visible_bone_value_count': int(sample_value_count),
        'reference_visible_bone_value_count': int(reference_value_count),
        'sample_visible_bone_edge_count': int(sample_edge_count),
        'reference_visible_bone_edge_count': int(reference_edge_count),
        'missing_reset_to_zero': bool(np.all(canonical[sic_missing] == 0.0)),
    }
    return canonical, sic_missing, state, metadata


def inverse_project_h36m17_window(generated, state, start, end):
    """把一个 SiC F16 输出逆变换到项目 H36M17 坐标。"""
    value = np.asarray(generated, dtype=np.float32)
    start = int(start)
    end = int(end)
    if value.shape != (end - start, 17, 3):
        raise ValueError('generated window shape does not match start/end')
    if state.get('transform_mode') != TRANSFORM_MODE:
        raise ValueError('coordinate transform state mode mismatch')
    scale_ratio = float(state['scale_ratio'])
    if not np.isfinite(scale_ratio) or scale_ratio <= 0.0:
        raise ValueError('coordinate transform scale ratio is invalid')
    query_root = np.asarray(state['query_root'][start:end], dtype=np.float32)
    prompt_root = np.asarray(state['prompt_root'][start:end], dtype=np.float32)
    sic = (value - prompt_root[:, None, :]) / scale_ratio + query_root[:, None, :]
    project = sic_to_project_joints(sic_axes_to_project(sic))
    if not np.isfinite(project).all():
        raise ValueError('inverse transformed output contains non-finite values')
    return np.asarray(project, dtype=np.float32)


__all__ = [
    'AXIS_MATRIX_PROJECT_TO_SIC',
    'PROJECT_TO_SIC_JOINTS',
    'SIC_TO_PROJECT_JOINTS',
    'TRANSFORM_MODE',
    'inverse_project_h36m17_window',
    'prepare_project_h36m17_sample',
    'project_axes_to_sic',
    'project_to_sic_joints',
    'sic_axes_to_project',
    'sic_to_project_joints',
]
