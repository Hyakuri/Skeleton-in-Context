import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import train as train_module
from lib.data.dataset import MotionDataset3D, select_deterministic_task_subset
from sars_adapter.train_subset import (
    OFFICIAL_TASKS,
    build_direct_run_config,
    build_effective_args,
    resolve_training_config,
)
from train import summarize_active_task_scores
from train import (
    build_training_identity,
    evaluate_motion_completion,
    validate_checkpoint_training_identity,
)
from sars_adapter.run_completion import validate_completion_checkpoint_identity


class DeterministicSubsetTest(unittest.TestCase):
    def test_same_seed_selects_same_files(self):
        files = [f'{index:08d}.pkl' for index in range(100)]
        first = select_deterministic_task_subset(files, 12, 42, 'MC', 'train')
        second = select_deterministic_task_subset(files, 12, 42, 'MC', 'train')
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)

    def test_different_seed_changes_selection(self):
        files = [f'{index:08d}.pkl' for index in range(100)]
        first = select_deterministic_task_subset(files, 12, 42, 'MC', 'train')
        second = select_deterministic_task_subset(files, 12, 43, 'MC', 'train')
        self.assertNotEqual(first, second)

    def test_test_split_is_never_limited(self):
        files = [f'{index:08d}.pkl' for index in range(20)]
        selected = select_deterministic_task_subset(files, 3, 42, 'MC', 'test')
        self.assertEqual(selected, files)


class TrainingConfigTest(unittest.TestCase):
    def _config(self, root, task_folders=None):
        task_folders = task_folders or ('H36M', 'AMASS', '3DPW_MC', 'H36M_FPE')
        for folder in task_folders:
            for split in ('train', 'test'):
                os.makedirs(os.path.join(root, folder, split), exist_ok=True)
        return {
            'source_config': os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml'
            ),
            'data_root': root,
            'checkpoint_root': os.path.join(root, 'checkpoints'),
            'run_name': 'unit_test',
            'tasks': list(OFFICIAL_TASKS),
            'train_sample_limits': {task: 4 for task in OFFICIAL_TASKS},
            'subset_seed': 42,
            'epochs': 1,
            'batch_size': 8,
            'test_batch_size': 8,
            'learning_rate': 0.0002,
            'num_workers': 0,
            'pin_memory': False,
            'prefetch_factor': 2,
            'persistent_workers': False,
            'no_eval': True,
            'seed': 42,
            'dry_run': True,
        }

    def test_resolved_config_keeps_official_four_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            config = resolve_training_config(self._config(root))
            args = build_effective_args(config)
        self.assertEqual(list(args.tasks), OFFICIAL_TASKS)
        self.assertEqual(args.train_sample_limits['MC'], 4)
        self.assertEqual(args.num_workers, 0)
        self.assertEqual(args.data.root_path, root)

    def test_direct_config_defaults_to_safe_mc_only_plan(self):
        config = build_direct_run_config()
        self.assertTrue(config['dry_run'])
        self.assertEqual(config['tasks'], ['MC'])
        self.assertEqual(config['train_sample_limits'], {'MC': None})

    def test_accepts_mc_only_training(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._config(root)
            config['tasks'] = ['MC']
            config['train_sample_limits'] = {'MC': 4}
            resolved = resolve_training_config(config)
        self.assertEqual(resolved['tasks'], ['MC'])
        self.assertEqual(resolved['task_scope'], 'single_task')

    def test_mc_only_requires_only_mc_dataset_directories(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._config(root, task_folders=('3DPW_MC',))
            config['tasks'] = ['MC']
            config['train_sample_limits'] = {'MC': None}
            resolved = resolve_training_config(config)
        self.assertEqual(resolved['tasks'], ['MC'])
        self.assertIsNone(resolved['train_sample_limits']['MC'])

    def test_rejects_unknown_or_out_of_order_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._config(root)
            config['tasks'] = ['MC', 'PE']
            config['train_sample_limits'] = {'MC': 4, 'PE': 4}
            with self.assertRaisesRegex(ValueError, 'official order'):
                resolve_training_config(config)

            config['tasks'] = ['JC']
            config['train_sample_limits'] = {'JC': 4}
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                resolve_training_config(config)

    def test_train_sample_limits_must_match_active_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._config(root)
            config['tasks'] = ['MC']
            with self.assertRaisesRegex(ValueError, 'active tasks'):
                resolve_training_config(config)

    def test_strict_resume_rejects_persistent_data_workers(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._config(root)
            config['num_workers'] = 2
            config['persistent_workers'] = True
            with self.assertRaisesRegex(ValueError, 'persistent_workers'):
                resolve_training_config(config)

            config['persistent_workers'] = False
            resolved = resolve_training_config(config)
        self.assertEqual(resolved['num_workers'], 2)
        self.assertFalse(resolved['persistent_workers'])

    def test_dataset_initialization_uses_configured_training_seed(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._config(root, task_folders=('3DPW_MC',))
            config['tasks'] = ['MC']
            config['train_sample_limits'] = {'MC': None}
            config['seed'] = 123
            train_dir = os.path.join(root, '3DPW_MC', 'train')
            with open(os.path.join(train_dir, 'sample.pkl'), 'wb') as stream:
                stream.write(b'sample')
            args = build_effective_args(resolve_training_config(config))
            MotionDataset3D(args, data_split='train')

        self.assertAlmostEqual(random.random(), random.Random(123).random())
        self.assertAlmostEqual(
            float(np.random.random()),
            float(np.random.RandomState(123).random_sample()),
        )


class ActiveTaskScoreTest(unittest.TestCase):
    def test_single_mc_score_is_valid_global_score(self):
        task_scores, global_score = summarize_active_task_scores(
            ['MC'], {'MC': 42.5}
        )
        self.assertEqual(task_scores, {'MC': 42.5})
        self.assertEqual(global_score, 42.5)

    def test_four_task_score_preserves_official_mean(self):
        values = {'PE': 10.0, 'MP': 20.0, 'MC': 30.0, 'FPE': 40.0}
        task_scores, global_score = summarize_active_task_scores(
            OFFICIAL_TASKS, values
        )
        self.assertEqual(task_scores, values)
        self.assertEqual(global_score, 25.0)


class TrainingIdentityTest(unittest.TestCase):
    def test_identity_binds_task_config_manifest_and_seed(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'effective_config.yaml')
            manifest_path = os.path.join(root, 'training_subset_manifest.json')
            with open(config_path, 'wb') as stream:
                stream.write(b'tasks: [MC]\n')
            with open(manifest_path, 'wb') as stream:
                stream.write(b'{"tasks":["MC"]}')
            identity = build_training_identity(
                config_path=config_path,
                subset_manifest_path=manifest_path,
                tasks=['MC'],
                subset_seed=42,
                training_seed=42,
            )
        self.assertEqual(identity['tasks'], ['MC'])
        self.assertEqual(identity['task_scope'], 'single_task')
        self.assertEqual(len(identity['effective_config_sha256']), 64)
        self.assertEqual(len(identity['subset_manifest_sha256']), 64)

    def test_resume_rejects_training_identity_mismatch(self):
        current = {
            'format': 'skeleton_in_context_training_identity',
            'version': 1,
            'tasks': ['MC'],
            'task_scope': 'single_task',
            'subset_seed': 42,
            'training_seed': 42,
            'effective_config_sha256': 'a' * 64,
            'subset_manifest_sha256': 'b' * 64,
        }
        validate_checkpoint_training_identity(
            {'training_identity': dict(current)}, current, strict=True
        )
        changed = dict(current)
        changed['tasks'] = ['PE', 'MP', 'MC', 'FPE']
        with self.assertRaisesRegex(ValueError, 'training identity mismatch'):
            validate_checkpoint_training_identity(
                {'training_identity': changed}, current, strict=True
            )

    def test_formal_completion_requires_matching_mc_only_config(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'effective_config.yaml')
            with open(config_path, 'wb') as stream:
                stream.write(b'tasks: [MC]\n')
            identity = {
                'format': 'skeleton_in_context_training_identity',
                'version': 1,
                'tasks': ['MC'],
                'task_scope': 'single_task',
                'effective_config_sha256': train_module._sha256_file(config_path),
            }
            validated = validate_completion_checkpoint_identity(
                {'training_identity': identity},
                config_path,
                policy='require_mc_only',
            )
            self.assertEqual(validated['tasks'], ['MC'])

            with open(config_path, 'ab') as stream:
                stream.write(b'epochs: 120\n')
            with self.assertRaisesRegex(ValueError, 'source config'):
                validate_completion_checkpoint_identity(
                    {'training_identity': identity},
                    config_path,
                    policy='require_mc_only',
                )

    def test_formal_completion_rejects_legacy_or_multitask_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'effective_config.yaml')
            with open(config_path, 'wb') as stream:
                stream.write(b'tasks: [MC]\n')
            with self.assertRaisesRegex(ValueError, 'training identity'):
                validate_completion_checkpoint_identity(
                    {}, config_path, policy='require_mc_only'
                )
            with self.assertRaisesRegex(ValueError, 'training identity'):
                validate_completion_checkpoint_identity({}, config_path)
            self.assertIsNone(validate_completion_checkpoint_identity(
                {}, config_path, policy='allow_legacy'
            ))

            identity = {
                'format': 'skeleton_in_context_training_identity',
                'version': 1,
                'tasks': list(OFFICIAL_TASKS),
                'task_scope': 'multi_task',
                'effective_config_sha256': train_module._sha256_file(config_path),
            }
            with self.assertRaisesRegex(ValueError, 'MC-only'):
                validate_completion_checkpoint_identity(
                    {'training_identity': identity},
                    config_path,
                    policy='require_mc_only',
                )


class MotionCompletionEvaluationTest(unittest.TestCase):
    class _Model:
        def eval(self):
            return self

        def __call__(self, prompt, query, epoch=None):
            target = query[:, 16:].clone()
            return target.clone(), target

    def test_cpu_evaluation_uses_query_device_for_mask_prefix(self):
        query_input = torch.ones((1, 16, 17, 3), dtype=torch.float32)
        query_input[:, :, 1:7] = 0
        query_target = torch.ones((1, 16, 17, 3), dtype=torch.float32)
        query = torch.cat([query_input, query_target], dim=1)
        prompt = query.clone()
        args = SimpleNamespace(
            backbone='SiC_dynamicTUP',
            task_to_flag={'MC': 3},
            data=SimpleNamespace(
                drop_ratios_MC=[0.4], num_joints=17, clip_len=16
            ),
        )
        with mock.patch.object(
            train_module.torch.cuda, 'is_available', return_value=False
        ):
            error, _ = evaluate_motion_completion(
                args, [(prompt, query, torch.tensor([3]))], self._Model()
            )
        self.assertEqual(error, 0.0)


class SingleTaskTrainLoopTest(unittest.TestCase):
    class _Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.prompt_list = {'MC': ['demo.pkl']}
            self.selection_manifest = {'MC': ['train.pkl']}
            self.global_idx_list = {'MC': [0]}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            sample = torch.zeros((32, 17, 3), dtype=torch.float32)
            return sample, sample, 3

    def test_mc_only_eval_saves_mc_checkpoints_without_pe_state(self):
        with tempfile.TemporaryDirectory() as root:
            config = TrainingConfigTest()._config(root)
            config['tasks'] = ['MC']
            config['train_sample_limits'] = {'MC': 1}
            config['no_eval'] = False
            args = build_effective_args(resolve_training_config(config))
            checkpoint_dir = os.path.join(root, 'run')
            opts = SimpleNamespace(
                checkpoint=checkpoint_dir,
                config=config['source_config'],
                resume='',
                evaluate='',
                seed=42,
            )
            dataset = self._Dataset()
            writer = mock.Mock()
            with mock.patch.object(
                train_module, 'MotionDataset3D', return_value=dataset
            ), mock.patch.object(
                train_module, 'load_backbone', return_value=torch.nn.Linear(1, 1)
            ), mock.patch.object(
                train_module, 'train_epoch'
            ), mock.patch.object(
                train_module, 'evaluate_motion_completion', return_value=(42.5, 'MC')
            ), mock.patch.object(
                train_module, 'save_checkpoint'
            ) as save_checkpoint, mock.patch.object(
                train_module.tensorboardX, 'SummaryWriter', return_value=writer
            ), mock.patch.object(
                train_module.torch.cuda, 'is_available', return_value=False
            ):
                train_module.train_with_config(args, opts)

        saved_names = [
            os.path.basename(call[0][0]) for call in save_checkpoint.call_args_list
        ]
        self.assertIn('latest_epoch.bin', saved_names)
        self.assertIn('best_epoch_MC.bin', saved_names)
        self.assertIn('best_epoch_all.bin', saved_names)
        self.assertNotIn('best_epoch_PE.bin', saved_names)
        calls_by_name = {
            os.path.basename(call[0][0]): call
            for call in save_checkpoint.call_args_list
        }
        latest = calls_by_name['latest_epoch.bin']
        for name in ('best_epoch_MC.bin', 'best_epoch_all.bin'):
            best = calls_by_name[name]
            self.assertEqual(best[0][2], latest[0][2])
            self.assertIsNotNone(best.kwargs.get('rng_state'))


if __name__ == '__main__':
    unittest.main()
