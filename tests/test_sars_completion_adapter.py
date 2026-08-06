import copy
import hashlib
import json
import pickle
import tempfile
import unittest
import os

import numpy as np
import torch

from sars_adapter.contracts import (
    load_completion_query,
    validate_completion_query,
)
from sars_adapter.windowing import (
    analyze_mask_compatibility,
    f64_windows,
    restore_observed_joints,
)
from sars_adapter.inference import (
    build_train_prompt_pool,
    build_train_prompt_provider,
    complete_f64_with_model,
)
from sars_adapter.run_completion import (
    EXTERNAL_PICKLE_PROTOCOL,
    _repository_identity,
    _resolve_demonstration_selection_policy,
    _validate_formal_coordinate_contract,
    build_direct_run_config,
    build_result_package,
)


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
    if isinstance(value, dict):
        return {key: _hashable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_hashable(child) for child in value]
    return value


def _query():
    package = {
        'format': 'sars_inter_external_completion_query',
        'version': 2,
        'dataset_profile': 'custom',
        'source_split': 'test',
        'completion_scope': 'all_split',
        'selection_manifest_hash': None,
        'sample_order': ['sample-a'],
        'frame_count': 64,
        'joint_count': 17,
        'missing_mask_semantics': 'True=missing',
        'coordinate_contract': {
            'skeleton': 'H36M17',
            'axis_order': ['x_lateral', 'y_depth', 'z_height'],
            'unit': 'dataset_normalized',
        },
        'temporal_metadata': {'frame_count': 64, 'fps': 30},
        'masked_keypoint': np.zeros((1, 64, 17, 3), dtype=np.float32),
        'missing_mask': np.zeros((1, 64, 17), dtype=bool),
        'metadata': {},
    }
    encoded = json.dumps(
        _hashable(package), sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    package['query_hash'] = hashlib.sha256(encoded).hexdigest()
    return package


def _rehash_query(package):
    package.pop('query_hash', None)
    encoded = json.dumps(
        _hashable(package), sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    package['query_hash'] = hashlib.sha256(encoded).hexdigest()
    return package


class CompletionContractTest(unittest.TestCase):
    def test_query_loader_rejects_evaluation_fields(self):
        package = _query()
        package['labels'] = [1]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + '/query.pkl'
            with open(path, 'wb') as stream:
                pickle.dump(package, stream)
            with self.assertRaisesRegex(ValueError, 'unsupported fields'):
                load_completion_query(path)

    def test_query_loader_rejects_nested_evaluation_fields_and_missing_keys(self):
        package = _query()
        package['metadata'] = {'rows': [{'clean_skeleton': 'private'}]}
        _rehash_query(package)
        with self.assertRaisesRegex(ValueError, 'private evaluation'):
            validate_completion_query(package)

        package = _query()
        package.pop('temporal_metadata')
        _rehash_query(package)
        with self.assertRaisesRegex(ValueError, 'missing required fields'):
            validate_completion_query(package)

        package = _query()
        package['metadata'] = {
            'rows': np.array(
                [(1,)], dtype=[('clean_skeleton', np.int32)]
            )
        }
        _rehash_query(package)
        with self.assertRaisesRegex(TypeError, 'JSON-like'):
            validate_completion_query(package)

    def test_query_loader_requires_float32_and_boolean_mask(self):
        package = _query()
        package['masked_keypoint'] = package['masked_keypoint'].astype(np.float64)
        _rehash_query(package)
        with self.assertRaisesRegex(ValueError, 'float32'):
            validate_completion_query(package)

        package = _query()
        package['missing_mask'] = package['missing_mask'].astype(np.uint8)
        _rehash_query(package)
        with self.assertRaisesRegex(ValueError, 'dtype must be bool'):
            validate_completion_query(package)

    def test_query_loader_accepts_explicit_boolean_mask(self):
        package = _query()
        package['missing_mask'][0, :, 3] = True
        package.pop('query_hash')
        encoded = json.dumps(
            _hashable(package), sort_keys=True, separators=(',', ':')
        ).encode('utf-8')
        package['query_hash'] = hashlib.sha256(encoded).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + '/query.pkl'
            with open(path, 'wb') as stream:
                pickle.dump(package, stream)
            loaded = load_completion_query(path)
        self.assertEqual(loaded['missing_mask'].dtype, np.bool_)
        self.assertTrue(loaded['missing_mask'][0, 0, 3])


class WindowingTest(unittest.TestCase):
    def test_f64_uses_four_non_overlapping_f16_windows(self):
        self.assertEqual(f64_windows(), [(0, 16), (16, 32), (32, 48), (48, 64)])

    def test_mask_compatibility_marks_temporal_and_boundary_joint_ood(self):
        mask = np.zeros((64, 17), dtype=bool)
        mask[:, 3:9] = True
        official = analyze_mask_compatibility(mask)
        self.assertTrue(official['official_mc_compatible'])

        mask[4:8, 3] = False
        mask[:, 0] = True
        ood = analyze_mask_compatibility(mask)
        self.assertFalse(ood['official_mc_compatible'])
        self.assertIn('time_varying_within_window', ood['reasons'])
        self.assertIn('unsupported_boundary_joint', ood['reasons'])

        mask = np.zeros((64, 17), dtype=bool)
        mask[:, 3] = True
        ood = analyze_mask_compatibility(mask)
        self.assertFalse(ood['official_mc_compatible'])
        self.assertIn('unsupported_missing_count', ood['reasons'])

    def test_restore_observed_joints_is_exact(self):
        masked = np.arange(64 * 17 * 3, dtype=np.float32).reshape(64, 17, 3)
        missing = np.zeros((64, 17), dtype=bool)
        missing[:, 4] = True
        generated = np.full_like(masked, -3.0)
        restored = restore_observed_joints(masked, generated, missing)
        np.testing.assert_array_equal(restored[~missing], masked[~missing])
        np.testing.assert_array_equal(restored[missing], generated[missing])


class _FakeDynamicModel(torch.nn.Module):
    def forward(self, prompt, query, epoch=None):
        batch = query.shape[0]
        prediction = torch.full(
            (batch, 16, 17, 3), 7.0, dtype=query.dtype, device=query.device
        )
        return prediction, query[:, 16:]


class _PromptTargetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = []

    def forward(self, prompt, query, epoch=None):
        self.queries.append(query.detach().cpu().numpy())
        return prompt[:, 16:], query[:, 16:]


class InferenceAdapterTest(unittest.TestCase):
    def test_formal_contract_requires_explicit_project_joint_order(self):
        contract = {
            'skeleton': 'H36M17',
            'axis_order': ['x_lateral', 'y_depth', 'z_height'],
            'unit': 'dataset_normalized',
        }
        with self.assertRaisesRegex(ValueError, 'joint_order'):
            _validate_formal_coordinate_contract(contract)
        contract['joint_order'] = 'h36m17_sars_inter_project_order'
        _validate_formal_coordinate_contract(contract)

    def test_prompt_policy_is_strictly_bound_to_transform_mode(self):
        self.assertEqual(
            _resolve_demonstration_selection_policy(
                'identity_h36m17', None
            ),
            'per_window',
        )
        self.assertEqual(
            _resolve_demonstration_selection_policy(
                'project_h36m17_prompt_aligned_v1', None
            ),
            'per_sample_fixed',
        )
        with self.assertRaisesRegex(ValueError, 'requires per_window'):
            _resolve_demonstration_selection_policy(
                'identity_h36m17', 'per_sample_fixed'
            )
        with self.assertRaisesRegex(ValueError, 'requires per_sample_fixed'):
            _resolve_demonstration_selection_policy(
                'project_h36m17_prompt_aligned_v1', 'per_window'
            )

    def test_repository_identity_supports_isolated_worktree(self):
        commit, dirty, worktree_hash = _repository_identity()
        self.assertEqual(len(commit), 40)
        self.assertIsInstance(dirty, bool)
        self.assertEqual(len(worktree_hash), 64)

    def test_direct_config_defaults_to_formal_coordinate_adapter(self):
        config = build_direct_run_config()
        self.assertEqual(
            config['coordinate_transform_mode'],
            'project_h36m17_prompt_aligned_v1',
        )
        self.assertEqual(
            config['demonstration_selection_policy'], 'per_sample_fixed'
        )

    def test_completion_uses_explicit_mask_and_restores_observed(self):
        masked = np.ones((1, 64, 17, 3), dtype=np.float32)
        missing = np.zeros((1, 64, 17), dtype=bool)
        missing[:, :, 3:9] = True
        masked[missing] = 0.0
        demo_input = np.full((16, 17, 3), 2.0, dtype=np.float32)
        demo_target = np.full((16, 17, 3), 3.0, dtype=np.float32)

        def prompt_provider(sample_index, window_index, window_mask):
            return demo_input, demo_target, {
                'source_split': 'train',
                'identity': f'demo-{sample_index}-{window_index}',
            }

        completed, metadata = complete_f64_with_model(
            model=_FakeDynamicModel(),
            masked_keypoint=masked,
            missing_mask=missing,
            prompt_provider=prompt_provider,
            device='cpu',
            mask_policy='strict_official_mc',
        )
        np.testing.assert_array_equal(completed[~missing], masked[~missing])
        np.testing.assert_array_equal(completed[missing], 7.0)
        self.assertEqual(len(metadata['windows']), 4)
        self.assertEqual(metadata['demonstration_source_splits'], ['train'])

    def test_prompt_provider_uses_only_deterministic_mc_train_files(self):
        with tempfile.TemporaryDirectory() as root:
            train_dir = os.path.join(root, '3DPW_MC', 'train')
            os.makedirs(train_dir)
            for index in range(2):
                payload = {
                    'data_input': np.full(
                        (16, 18, 3), float(index + 1), dtype=np.float32
                    ),
                    'data_label': np.full(
                        (16, 18, 3), float(index + 2), dtype=np.float32
                    ),
                }
                with open(os.path.join(train_dir, '{}.pkl'.format(index)), 'wb') as stream:
                    pickle.dump(payload, stream)
            provider = build_train_prompt_provider(
                data_root=root,
                source_config=os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'configs',
                    'default.yaml',
                ),
                seed=42,
            )
            first = provider(0, 0, np.zeros((16, 17), dtype=bool))
            second = provider(0, 0, np.zeros((16, 17), dtype=bool))

        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[2]['source_split'], 'train')
        self.assertEqual(first[2]['identity'], second[2]['identity'])
        self.assertEqual(first[0].shape, (16, 17, 3))
        self.assertEqual(first[2]['prompt_pool_file_count'], 2)
        self.assertEqual(
            len(first[2]['prompt_pool_manifest_sha256']), 64
        )
        self.assertEqual(len(first[2]['source_config_sha256']), 64)
        self.assertEqual(len(first[2]['selection_key_sha256']), 64)

    def test_prompt_pool_manifest_changes_when_file_content_changes(self):
        with tempfile.TemporaryDirectory() as root:
            train_dir = os.path.join(root, '3DPW_MC', 'train')
            os.makedirs(train_dir)
            path = os.path.join(train_dir, 'same-size.pkl')
            with open(path, 'wb') as stream:
                stream.write(b'content-a')
            common = {
                'data_root': root,
                'source_config': os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'configs', 'default.yaml',
                ),
            }
            first = build_train_prompt_provider(**common)
            with open(path, 'wb') as stream:
                stream.write(b'content-b')
            second = build_train_prompt_provider(**common)

        self.assertNotEqual(
            first.prompt_pool_manifest_sha256,
            second.prompt_pool_manifest_sha256,
        )

    def test_prompt_provider_rejects_selected_demo_content_drift(self):
        with tempfile.TemporaryDirectory() as root:
            train_dir = os.path.join(root, '3DPW_MC', 'train')
            os.makedirs(train_dir)
            path = os.path.join(train_dir, 'demo.pkl')
            with open(path, 'wb') as stream:
                pickle.dump(np.zeros((16, 17, 3), dtype=np.float32), stream)
            source_config = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'configs', 'default.yaml',
            )
            pool = build_train_prompt_pool(root, source_config)
            provider = build_train_prompt_provider(
                root,
                source_config,
                prompt_pool=pool,
                selection_keys=['sample-key'],
            )
            with open(path, 'wb') as stream:
                pickle.dump(np.ones((16, 17, 3), dtype=np.float32), stream)

            with self.assertRaisesRegex(RuntimeError, 'prompt.*changed'):
                provider(0, 0, np.zeros((16, 17), dtype=bool))

    def test_prompt_provider_can_reuse_one_train_demo_for_all_windows(self):
        with tempfile.TemporaryDirectory() as root:
            train_dir = os.path.join(root, '3DPW_MC', 'train')
            os.makedirs(train_dir)
            for index in range(3):
                payload = {
                    'data_input': np.full(
                        (16, 18, 3), float(index + 1), dtype=np.float32
                    ),
                    'data_label': np.full(
                        (16, 18, 3), float(index + 2), dtype=np.float32
                    ),
                }
                with open(os.path.join(train_dir, '{}.pkl'.format(index)), 'wb') as stream:
                    pickle.dump(payload, stream)
            provider = build_train_prompt_provider(
                data_root=root,
                source_config=os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'configs',
                    'default.yaml',
                ),
                seed=42,
                selection_policy='per_sample_fixed',
            )
            identities = [
                provider(0, window_index, np.zeros((16, 17), dtype=bool))[2][
                    'identity'
                ]
                for window_index in range(4)
            ]

        self.assertEqual(len(set(identities)), 1)

    def test_prompt_selection_key_is_invariant_to_query_order(self):
        with tempfile.TemporaryDirectory() as root:
            train_dir = os.path.join(root, '3DPW_MC', 'train')
            os.makedirs(train_dir)
            for index in range(5):
                payload = {
                    'data_input': np.full(
                        (16, 18, 3), float(index + 1), dtype=np.float32
                    ),
                    'data_label': np.full(
                        (16, 18, 3), float(index + 2), dtype=np.float32
                    ),
                }
                with open(os.path.join(train_dir, '{}.pkl'.format(index)), 'wb') as stream:
                    pickle.dump(payload, stream)
            common = {
                'data_root': root,
                'source_config': os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'configs',
                    'default.yaml',
                ),
                'seed': 42,
                'selection_policy': 'per_sample_fixed',
            }
            first = build_train_prompt_provider(
                selection_keys=['input-a', 'input-b'], **common
            )
            second = build_train_prompt_provider(
                selection_keys=['input-b', 'input-a'], **common
            )
            mask = np.zeros((16, 17), dtype=bool)
            first_identity = first(0, 0, mask)[2]['identity']
            second_identity = second(1, 0, mask)[2]['identity']

        self.assertEqual(first_identity, second_identity)

    def test_direct_inference_rejects_non_boolean_mask(self):
        masked = np.ones((1, 64, 17, 3), dtype=np.float32)
        missing = np.zeros((1, 64, 17), dtype=np.uint8)

        def prompt_provider(sample_index, window_index, window_mask):
            demo = np.ones((16, 17, 3), dtype=np.float32)
            return demo, demo, {
                'source_split': 'train',
                'identity': 'demo',
            }

        with self.assertRaisesRegex(ValueError, 'dtype must be bool'):
            complete_f64_with_model(
                model=_FakeDynamicModel(),
                masked_keypoint=masked,
                missing_mask=missing,
                prompt_provider=prompt_provider,
                device='cpu',
            )

    def test_formal_coordinate_mode_uses_one_prompt_and_shared_transform(self):
        masked = np.zeros((1, 64, 17, 3), dtype=np.float32)
        for joint in range(17):
            masked[0, :, joint] = [joint * 0.1, joint * 0.03, joint * 0.07]
        missing = np.zeros((1, 64, 17), dtype=bool)
        missing[:, :, 7:] = True
        masked[missing] = 0.0

        demo_input = np.zeros((16, 17, 3), dtype=np.float32)
        demo_target = np.zeros((16, 17, 3), dtype=np.float32)
        for joint in range(17):
            demo_input[:, joint] = [joint * 0.05, joint * 0.04, joint * 0.02]
            demo_target[:, joint] = [joint * 0.06, joint * 0.05, joint * 0.03]
        calls = []

        def prompt_provider(sample_index, window_index, window_mask):
            calls.append((sample_index, window_index))
            return demo_input, demo_target, {
                'source_split': 'train',
                'identity': 'fixed-demo',
                'sha256': 'a' * 64,
                'selection_policy': 'per_sample_fixed',
            }

        model = _PromptTargetModel()
        completed, metadata = complete_f64_with_model(
            model=model,
            masked_keypoint=masked,
            missing_mask=missing,
            prompt_provider=prompt_provider,
            device='cpu',
            mask_policy='allow_ood_explicit',
            coordinate_transform_mode='project_h36m17_prompt_aligned_v1',
        )

        self.assertEqual(calls, [(0, 0)])
        self.assertEqual(len(model.queries), 4)
        self.assertTrue(np.isfinite(completed).all())
        np.testing.assert_array_equal(completed[~missing], masked[~missing])
        transform = metadata['sample_transforms'][0]
        self.assertEqual(
            transform['transform_mode'],
            'project_h36m17_prompt_aligned_v1',
        )
        self.assertTrue(transform['missing_reset_to_zero'])
        self.assertEqual(
            {row['demonstration']['identity'] for row in metadata['windows']},
            {'fixed-demo'},
        )
        self.assertEqual(len(metadata['boundary_diagnostics']), 3)
        self.assertGreater(
            metadata['boundary_diagnostics'][0]['missing_joint_count'], 0
        )

    def test_result_package_binds_query_and_records_runtime_policy(self):
        query = _query()
        completed = np.ones((1, 64, 17, 3), dtype=np.float32)
        package = build_result_package(
            query=query,
            completed_keypoint=completed,
            repository_commit='361e1c0b9552baa8510e00dbb33629debfd66873',
            checkpoint_identity='a' * 64,
            runtime_metadata={'total_seconds': 1.0, 'device': 'cpu'},
            method_metadata={
                'inference_task': 'joint_completion',
                'demonstration_source_splits': ['train'],
                'window_policy': 'non_overlapping_f16',
            },
        )
        self.assertEqual(package['query_hash'], query['query_hash'])

    def test_external_result_uses_pickle_protocol_four(self):
        query = _query()
        package = build_result_package(
            query=query,
            completed_keypoint=query['masked_keypoint'],
            repository_commit='commit',
            checkpoint_identity='checkpoint',
            runtime_metadata={'total_seconds': 1.0, 'device': 'cpu'},
            method_metadata={
                'inference_task': 'joint_completion',
                'demonstration_source_splits': ['train'],
            },
        )
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'result.pkl')
            with open(path, 'wb') as stream:
                pickle.dump(package, stream, protocol=EXTERNAL_PICKLE_PROTOCOL)
            with open(path, 'rb') as stream:
                self.assertEqual(stream.read(2), b'\x80\x04')
        self.assertEqual(package['sample_order'], ['sample-a'])
        self.assertEqual(len(package['result_hash']), 64)


if __name__ == '__main__':
    unittest.main()
