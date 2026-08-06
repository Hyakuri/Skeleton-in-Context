import unittest

import numpy as np

from sars_adapter.coordinate_adapter import (
    PROJECT_TO_SIC_JOINTS,
    inverse_project_h36m17_window,
    prepare_project_h36m17_sample,
    project_axes_to_sic,
    project_to_sic_joints,
    sic_axes_to_project,
    sic_to_project_joints,
)


SIC_EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)


def _sic_pose():
    pose = np.zeros((17, 3), dtype=np.float32)
    pose[1] = [0.2, -0.1, 0.0]
    pose[2] = [0.2, -0.6, 0.0]
    pose[3] = [0.2, -1.1, -0.1]
    pose[4] = [-0.2, -0.1, 0.0]
    pose[5] = [-0.2, -0.6, 0.0]
    pose[6] = [-0.2, -1.1, -0.1]
    pose[7] = [0.0, 0.2, 0.0]
    pose[8] = [0.0, 0.5, 0.0]
    pose[9] = [0.0, 0.7, 0.0]
    pose[10] = [0.0, 0.9, 0.0]
    pose[11] = [-0.25, 0.5, 0.0]
    pose[12] = [-0.55, 0.45, 0.0]
    pose[13] = [-0.85, 0.4, 0.0]
    pose[14] = [0.25, 0.5, 0.0]
    pose[15] = [0.55, 0.45, 0.0]
    pose[16] = [0.85, 0.4, 0.0]
    return pose


class JointAndAxisContractTest(unittest.TestCase):
    def test_joint_and_mask_permutation_is_exact_and_self_inverse(self):
        project = np.zeros((2, 17, 3), dtype=np.float32)
        project[..., 0] = np.arange(17, dtype=np.float32)
        mask = np.zeros((2, 17), dtype=bool)
        mask[:, 1] = True
        mask[:, 14] = True

        sic = project_to_sic_joints(project)
        sic_mask = project_to_sic_joints(mask)

        np.testing.assert_array_equal(
            sic[..., 0],
            np.tile(np.asarray(PROJECT_TO_SIC_JOINTS)[None, :], (2, 1)),
        )
        self.assertTrue(np.all(sic_mask[:, 4]))
        self.assertTrue(np.all(sic_mask[:, 11]))
        np.testing.assert_array_equal(sic_to_project_joints(sic), project)
        np.testing.assert_array_equal(sic_to_project_joints(sic_mask), mask)

    def test_axis_rotation_is_right_handed_and_exactly_reversible(self):
        project = np.asarray(
            [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32
        )
        sic = project_axes_to_sic(project)
        np.testing.assert_array_equal(
            sic,
            np.asarray([[1.0, 3.0, -2.0], [-4.0, -6.0, -5.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(sic_axes_to_project(sic), project)


class SampleAlignmentTest(unittest.TestCase):
    def _sample(self):
        prompt = np.repeat(_sic_pose()[None, ...], 16, axis=0)
        prompt[:, 0] += np.asarray([0.1, 0.25, -0.05], dtype=np.float32)
        prompt[:, 1:] += prompt[:, 0:1]

        root = np.zeros((64, 3), dtype=np.float32)
        root[:, 0] = np.linspace(-0.2, 0.2, 64)
        root[:, 2] = np.linspace(0.1, -0.1, 64)
        sic = np.repeat(_sic_pose()[None, ...], 64, axis=0) * 2.0
        sic[:, 1:] += root[:, None, :]
        sic[:, 0] = root
        project = sic_to_project_joints(sic_axes_to_project(sic))

        missing = np.zeros((64, 17), dtype=bool)
        missing[:, 7:] = True
        masked = np.array(project, copy=True)
        masked[missing] = 0.0
        return prompt, project, masked, missing

    def test_f64_transform_uses_one_scale_and_inverse_roundtrips(self):
        prompt, project, masked, missing = self._sample()

        canonical, canonical_mask, state, metadata = (
            prepare_project_h36m17_sample(masked, missing, prompt)
        )

        self.assertEqual(canonical.shape, (64, 17, 3))
        self.assertEqual(canonical_mask.shape, (64, 17))
        self.assertAlmostEqual(metadata['scale_ratio'], 0.5, places=6)
        self.assertGreater(metadata['sample_visible_bone_value_count'], 0)
        self.assertEqual(state['scale_ratio'], metadata['scale_ratio'])
        self.assertTrue(np.all(canonical[canonical_mask] == 0.0))

        restored = np.empty_like(project)
        for start in range(0, 64, 16):
            generated = np.repeat(prompt, 4, axis=0)[start:start + 16]
            restored[start:start + 16] = inverse_project_h36m17_window(
                generated, state, start, start + 16
            )

        np.testing.assert_allclose(restored, project, atol=1e-6, rtol=0.0)

    def test_missing_root_anchor_is_rejected(self):
        prompt, _, masked, missing = self._sample()
        missing[:, [0, 1, 4]] = True
        masked[missing] = 0.0

        with self.assertRaisesRegex(ValueError, 'root anchor'):
            prepare_project_h36m17_sample(masked, missing, prompt)


if __name__ == '__main__':
    unittest.main()
