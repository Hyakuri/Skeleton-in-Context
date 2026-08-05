import os
import tempfile
import unittest

from lib.data.dataset import select_deterministic_task_subset
from sars_adapter.train_subset import (
    OFFICIAL_TASKS,
    build_effective_args,
    resolve_training_config,
)


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
    def _config(self, root):
        for folder in ('H36M', 'AMASS', '3DPW_MC', 'H36M_FPE'):
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

    def test_rejects_task_removal(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._config(root)
            config['tasks'] = ['MC']
            with self.assertRaisesRegex(ValueError, 'official order'):
                resolve_training_config(config)


if __name__ == '__main__':
    unittest.main()
