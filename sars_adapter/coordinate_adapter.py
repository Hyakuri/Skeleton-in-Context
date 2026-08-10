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
ANCHOR_POLICY = 'paired_visible_semantic_anchor'


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


def _interpolate_trajectory(values, available):
    trajectory = np.asarray(values, dtype=np.float32).copy()
    available = np.asarray(available, dtype=bool)
    frame_count = trajectory.shape[0]
    known = np.flatnonzero(available)
    if known.size == 0:
        raise ValueError('trajectory has no visible anchor')
    frame_axis = np.arange(frame_count, dtype=np.float64)
    for coordinate in range(3):
        trajectory[:, coordinate] = np.interp(
            frame_axis,
            known.astype(np.float64),
            trajectory[known, coordinate],
        ).astype(np.float32)
    return trajectory


def _anchor_record(mode, joint_ids, available, source_counts=None,
                   interpolated_joint_frame_count=None):
    available = np.asarray(available, dtype=bool)
    record = {
        'anchor_mode': str(mode),
        'anchor_joint_ids': [int(value) for value in joint_ids],
        'observed_anchor_frame_count': int(np.sum(available)),
        'interpolated_anchor_frame_count': int(np.sum(~available)),
    }
    if source_counts:
        record['anchor_source_counts'] = {
            str(key): int(value) for key, value in source_counts.items()
        }
    if interpolated_joint_frame_count is not None:
        record['interpolated_joint_frame_count'] = int(
            interpolated_joint_frame_count
        )
    return record


def _paired_anchor_trajectories(keypoint, missing_mask, prompt):
    """只用 query 可见点构建与 train prompt 语义对应的 F64 anchor。"""
    frame_count = keypoint.shape[0]

    # 优先保持 pelvis 语义；pelvis 缺帧时，双髋中点仍表示同一人体中心。
    root = np.zeros((frame_count, 3), dtype=np.float32)
    pelvis_visible = ~missing_mask[:, 0]
    hip_visible = missing_mask[:, 0] & ~missing_mask[:, 1] & ~missing_mask[:, 4]
    root[pelvis_visible] = keypoint[pelvis_visible, 0]
    root[hip_visible] = (
        keypoint[hip_visible, 1] + keypoint[hip_visible, 4]
    ) * 0.5
    root_available = pelvis_visible | hip_visible
    if np.any(root_available):
        query_anchor = _interpolate_trajectory(root, root_available)
        metadata = _anchor_record(
            'pelvis_or_hip_midpoint',
            [0, 1, 4],
            root_available,
            source_counts={
                'pelvis': np.sum(pelvis_visible),
                'hip_midpoint': np.sum(hip_visible),
                'temporal_interpolation': np.sum(~root_available),
            },
        )
        return query_anchor, np.asarray(prompt[:, 0], dtype=np.float32), metadata

    for mode, joint_id in (('center_torso', 7), ('upper_torso', 8)):
        available = ~missing_mask[:, joint_id]
        if np.any(available):
            query_anchor = _interpolate_trajectory(
                keypoint[:, joint_id], available
            )
            metadata = _anchor_record(mode, [joint_id], available)
            return (
                query_anchor,
                np.asarray(prompt[:, joint_id], dtype=np.float32),
                metadata,
            )

    shoulder_ids = [11, 14]
    shoulder_available = ~(
        missing_mask[:, shoulder_ids[0]]
        | missing_mask[:, shoulder_ids[1]]
    )
    if np.any(shoulder_available):
        shoulder_midpoint = (
            keypoint[:, shoulder_ids[0]] + keypoint[:, shoulder_ids[1]]
        ) * 0.5
        query_anchor = _interpolate_trajectory(
            shoulder_midpoint, shoulder_available
        )
        prompt_anchor = np.mean(prompt[:, shoulder_ids], axis=1)
        metadata = _anchor_record(
            'shoulder_midpoint', shoulder_ids, shoulder_available
        )
        return query_anchor, np.asarray(prompt_anchor, dtype=np.float32), metadata

    # 最后仅使用固定 joint 集合；每个 joint 独立插值后再取质心，避免逐帧更换语义。
    visible_joint_ids = np.flatnonzero(np.any(~missing_mask, axis=0))
    if visible_joint_ids.size < 3:
        raise ValueError('sample has no visible semantic anchor')
    joint_trajectories = []
    interpolated_joint_frames = 0
    for joint_id in visible_joint_ids.tolist():
        available = ~missing_mask[:, joint_id]
        joint_trajectories.append(
            _interpolate_trajectory(keypoint[:, joint_id], available)
        )
        interpolated_joint_frames += int(np.sum(~available))
    query_anchor = np.mean(
        np.stack(joint_trajectories, axis=1), axis=1
    ).astype(np.float32)
    prompt_anchor = np.mean(prompt[:, visible_joint_ids], axis=1)
    all_visible = np.all(~missing_mask[:, visible_joint_ids], axis=1)
    metadata = _anchor_record(
        'fixed_visible_centroid',
        visible_joint_ids.tolist(),
        all_visible,
        interpolated_joint_frame_count=interpolated_joint_frames,
    )
    return query_anchor, np.asarray(prompt_anchor, dtype=np.float32), metadata


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
    prompt_f64 = np.tile(prompt, (4, 1, 1))
    query_anchor, prompt_anchor, anchor_metadata = _paired_anchor_trajectories(
        sic_masked, sic_missing, prompt_f64
    )
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
        (sic_masked - query_anchor[:, None, :]) * scale_ratio
        + prompt_anchor[:, None, :]
    ).astype(np.float32)
    canonical[sic_missing] = 0.0
    state = {
        'transform_mode': TRANSFORM_MODE,
        'query_anchor': query_anchor,
        'prompt_anchor': prompt_anchor,
        'scale_ratio': scale_ratio,
    }
    metadata = {
        'transform_mode': TRANSFORM_MODE,
        'transform_version': 2,
        'project_to_sic_joint_permutation': list(PROJECT_TO_SIC_JOINTS),
        'axis_matrix_project_to_sic': AXIS_MATRIX_PROJECT_TO_SIC.tolist(),
        'anchor_policy': ANCHOR_POLICY,
        'query_anchor_sha256': _array_sha256(query_anchor),
        'prompt_anchor_sha256': _array_sha256(prompt_anchor),
        'sample_scale': sample_scale,
        'reference_scale': reference_scale,
        'scale_ratio': scale_ratio,
        'sample_visible_bone_value_count': int(sample_value_count),
        'reference_visible_bone_value_count': int(reference_value_count),
        'sample_visible_bone_edge_count': int(sample_edge_count),
        'reference_visible_bone_edge_count': int(reference_edge_count),
        'missing_reset_to_zero': bool(np.all(canonical[sic_missing] == 0.0)),
    }
    metadata.update(anchor_metadata)
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
    query_anchor = np.asarray(
        state['query_anchor'][start:end], dtype=np.float32
    )
    prompt_anchor = np.asarray(
        state['prompt_anchor'][start:end], dtype=np.float32
    )
    sic = (
        (value - prompt_anchor[:, None, :]) / scale_ratio
        + query_anchor[:, None, :]
    )
    project = sic_to_project_joints(sic_axes_to_project(sic))
    if not np.isfinite(project).all():
        raise ValueError('inverse transformed output contains non-finite values')
    return np.asarray(project, dtype=np.float32)


__all__ = [
    'AXIS_MATRIX_PROJECT_TO_SIC',
    'PROJECT_TO_SIC_JOINTS',
    'SIC_TO_PROJECT_JOINTS',
    'TRANSFORM_MODE',
    'ANCHOR_POLICY',
    'inverse_project_h36m17_window',
    'prepare_project_h36m17_sample',
    'project_axes_to_sic',
    'project_to_sic_joints',
    'sic_axes_to_project',
    'sic_to_project_joints',
]
