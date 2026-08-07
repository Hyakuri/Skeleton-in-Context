from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from sars_adapter.contracts import _query_hash
from sars_adapter.run_completion import _package_hash, _repository_identity
from sars_adapter.run_completion_series import run_completion_series
from sars_adapter.series_contracts import (
    build_run_plan,
    load_run_plan,
    validate_run_plan,
)


def _query(sample_id, offset=0.0):
    masked = np.full((1, 64, 17, 3), float(offset), dtype=np.float32)
    missing = np.zeros((1, 64, 17), dtype=bool)
    missing[:, 16:48, 8:] = True
    masked[missing] = 0.0
    package = {
        'format': 'sars_inter_external_completion_query',
        'version': 2,
        'dataset_profile': 'custom_worksite',
        'source_split': 'test',
        'completion_scope': 'all_split',
        'selection_manifest_hash': None,
        'sample_order': [sample_id],
        'frame_count': 64,
        'joint_count': 17,
        'missing_mask_semantics': 'True=missing',
        'coordinate_contract': {
            'skeleton': 'H36M17',
            'joint_order': 'h36m17_sars_inter_project_order',
            'axis_order': ['x_lateral', 'y_depth', 'z_height'],
            'unit': 'project_native',
        },
        'temporal_metadata': {'target_frame_count': 64},
        'masked_keypoint': masked,
        'missing_mask': missing,
        'metadata': {'mask': 'upper'},
    }
    package['query_hash'] = _query_hash(package)
    return package


class CompletionSeriesTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = self.tempdir.name
        self.output_root = os.path.join(self.root, 'results')
        self.checkpoint_path = os.path.join(self.root, 'checkpoint.bin')
        self.source_config = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml'
        )
        self.data_root = os.path.join(self.root, 'data')
        train_dir = os.path.join(self.data_root, '3DPW_MC', 'train')
        os.makedirs(train_dir)
        with open(self.checkpoint_path, 'wb') as stream:
            stream.write(b'checkpoint-for-series-test')
        with open(os.path.join(train_dir, 'demo.pkl'), 'wb') as stream:
            stream.write(b'demonstration-content')

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_query(self, name, offset=0.0):
        path = os.path.join(self.root, '{}.pkl'.format(name))
        package = _query(name, offset=offset)
        with open(path, 'wb') as stream:
            pickle.dump(package, stream, protocol=4)
        return path, package

    def _job(self, name, result_path=None, offset=0.0, mask='upper'):
        query_path, package = self._write_query(name, offset=offset)
        return {
            'job_id': name,
            'dataset_profile': 'custom_worksite',
            'mask': mask,
            'model': None,
            'source_split': 'test',
            'completion_scope': 'all_split',
            'query_path': query_path,
            'query_hash': package['query_hash'],
            'result_relative_path': result_path or '{}.pkl'.format(name),
            'consumer_models': ['stgcnpp', 'ctrgcn'],
            'expected_sample_count': 1,
        }

    def _method_profile(self):
        with open(self.checkpoint_path, 'rb') as stream:
            checkpoint_sha256 = hashlib.sha256(stream.read()).hexdigest()
        return {
            'method_name': 'skeleton_in_context',
            'repository_url': (
                'https://github.com/Hyakuri/Skeleton-in-Context'
            ),
            'repository_commit': _repository_identity()[0],
            'upstream_repository_url': (
                'https://github.com/fanglaosi/Skeleton-in-Context'
            ),
            'upstream_repository_commit': (
                '361e1c0b9552baa8510e00dbb33629debfd66873'
            ),
            'checkpoint_sha256': checkpoint_sha256,
        }

    def _write_plan(self, jobs):
        plan = build_run_plan(
            jobs, method_profile=self._method_profile()
        )
        path = os.path.join(self.root, 'run_plan.json')
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
        return path, plan

    def _config(self, plan_path, **overrides):
        config = {
            'plan_path': plan_path,
            'output_root': self.output_root,
            'checkpoint_path': self.checkpoint_path,
            'source_config': self.source_config,
            'checkpoint_identity_policy': 'allow_legacy',
            'data_root': self.data_root,
            'device': 'cpu',
            'dry_run': False,
            'resume': True,
            'strict': True,
            'continue_on_error': False,
            'mask_policy': 'allow_ood_explicit',
            'demonstration_seed': 42,
            'demonstration_selection_policy': 'per_sample_fixed',
            'coordinate_transform_mode': 'project_h36m17_prompt_aligned_v1',
            'require_clean_repository': False,
        }
        config.update(overrides)
        return config

    def test_formal_preflight_rejects_legacy_checkpoint_before_runner(self):
        torch.save({'model_pos': {}}, self.checkpoint_path)
        plan_path, _ = self._write_plan([self._job('legacy-identity')])
        runner = mock.Mock(side_effect=AssertionError('runner must not start'))
        with self.assertRaisesRegex(ValueError, 'training identity'):
            run_completion_series(
                self._config(
                    plan_path,
                    checkpoint_identity_policy='require_mc_only',
                ),
                completion_runner=runner,
            )
        runner.assert_not_called()

    def test_formal_preflight_rejects_wrong_effective_config(self):
        torch.save({
            'training_identity': {
                'format': 'skeleton_in_context_training_identity',
                'version': 1,
                'tasks': ['MC'],
                'task_scope': 'single_task',
                'subset_seed': 42,
                'training_seed': 42,
                'effective_config_sha256': '0' * 64,
                'subset_manifest_sha256': '1' * 64,
            },
            'model_pos': {},
        }, self.checkpoint_path)
        plan_path, _ = self._write_plan([self._job('wrong-config')])
        runner = mock.Mock(side_effect=AssertionError('runner must not start'))
        with self.assertRaisesRegex(ValueError, 'source config'):
            run_completion_series(
                self._config(
                    plan_path,
                    checkpoint_identity_policy='require_mc_only',
                ),
                completion_runner=runner,
            )
        runner.assert_not_called()

    @staticmethod
    def _fake_prompt_provider(selection_records):
        def factory(*args, **kwargs):
            selection_records.append(tuple(kwargs.get('selection_keys') or []))

            def provider(sample_index, window_index, window_mask):
                demo = np.zeros((16, 17, 3), dtype=np.float32)
                return demo, demo, {'source_split': 'train', 'identity': 'demo'}

            provider.prompt_pool_file_count = 1
            provider.prompt_pool_manifest_sha256 = 'd' * 64
            return provider

        return factory

    @staticmethod
    def _fake_completion(**kwargs):
        masked = np.asarray(kwargs['masked_keypoint'], dtype=np.float32)
        return np.array(masked, copy=True), {
            'window_policy': 'non_overlapping_f16',
            'windows': [],
            'mask_policy': kwargs['mask_policy'],
            'sample_transforms': [],
            'boundary_diagnostics': [],
            'observed_restoration_max_abs_error': 0.0,
        }

    def test_plan_rejects_recursive_private_fields(self):
        job = self._job('private')
        job['metadata'] = {'nested': {'gt_label': 3}}
        with self.assertRaisesRegex(ValueError, 'private'):
            build_run_plan([job])

        profile = self._method_profile()
        profile['trigger_manifest_path'] = 'private-trigger.json'
        with self.assertRaisesRegex(ValueError, 'private'):
            build_run_plan([self._job('private-profile')], profile)

    def test_plan_rejects_duplicate_job_result_and_identity(self):
        first = self._job('first', result_path='same.pkl')
        second = self._job(
            'second', result_path='same.pkl', mask='bottom'
        )
        with self.assertRaisesRegex(ValueError, 'result_relative_path'):
            build_run_plan([first, second], self._method_profile())

        second = self._job(
            'second', result_path='SAME.pkl', mask='bottom'
        )
        with self.assertRaisesRegex(ValueError, 'result_relative_path'):
            build_run_plan([first, second], self._method_profile())

        second = self._job('second', result_path='second.pkl')
        with self.assertRaisesRegex(ValueError, 'identity'):
            build_run_plan([first, second], self._method_profile())

        second = self._job('second', mask='bottom')
        second['job_id'] = first['job_id']
        with self.assertRaisesRegex(ValueError, 'job_id'):
            build_run_plan([first, second], self._method_profile())

    def test_plan_rejects_unsafe_result_paths(self):
        for unsafe in ('../result.pkl', '/absolute/result.pkl', 'C:\\result.pkl'):
            with self.subTest(path=unsafe):
                with self.assertRaisesRegex(ValueError, 'result_relative_path'):
                    build_run_plan(
                        [self._job('unsafe', result_path=unsafe)],
                        self._method_profile(),
                    )

    def test_plan_rejects_query_hash_mismatch(self):
        job = self._job('mismatch')
        job['query_hash'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'query_hash'):
            build_run_plan([job], self._method_profile())

    def test_plan_hash_is_stable_and_verified_on_load(self):
        plan_path, plan = self._write_plan([self._job('stable')])
        self.assertEqual(
            plan,
            build_run_plan(plan['jobs'], plan['method_profile']),
        )
        self.assertEqual(load_run_plan(plan_path)['plan_hash'], plan['plan_hash'])

        plan['method_profile']['method_name'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'plan_hash'):
            validate_run_plan(plan)

        noncanonical = self._method_profile()
        noncanonical['method_name'] = ' skeleton_in_context '
        with self.assertRaisesRegex(ValueError, 'canonical'):
            build_run_plan([self._job('noncanonical')], noncanonical)

    def test_formal_execution_rejects_dirty_repository(self):
        plan_path, plan = self._write_plan([self._job('dirty')])
        commit = plan['method_profile']['repository_commit']
        with mock.patch(
            'sars_adapter.run_completion._repository_identity',
            return_value=(commit, True, 'f' * 64),
        ):
            with self.assertRaisesRegex(ValueError, 'clean repository'):
                run_completion_series(self._config(
                    plan_path, require_clean_repository=True
                ))

    def test_execution_rejects_plan_for_different_adapter_commit(self):
        plan_path, plan = self._write_plan([self._job('wrong-adapter')])
        plan['method_profile']['repository_commit'] = '0' * 40
        plan = build_run_plan(plan['jobs'], plan['method_profile'])
        with open(plan_path, 'w', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)

        with self.assertRaisesRegex(ValueError, 'adapter.*commit'):
            run_completion_series(self._config(plan_path))

    def test_execution_rejects_plan_for_different_upstream_commit(self):
        plan_path, plan = self._write_plan([self._job('wrong-upstream')])
        plan['method_profile']['upstream_repository_commit'] = '0' * 40
        plan = build_run_plan(plan['jobs'], plan['method_profile'])
        with open(plan_path, 'w', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)

        with self.assertRaisesRegex(ValueError, 'upstream.*commit'):
            run_completion_series(self._config(plan_path))

    def test_execution_rejects_plan_for_different_checkpoint_before_model_load(self):
        plan_path, plan = self._write_plan([self._job('wrong-checkpoint')])
        plan['method_profile']['checkpoint_sha256'] = '0' * 64
        plan = build_run_plan(plan['jobs'], plan['method_profile'])
        with open(plan_path, 'w', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)

        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            run_completion_series(
                self._config(plan_path),
                completion_runner=mock.Mock(
                    side_effect=AssertionError('completion must not start')
                ),
            )

    def test_dry_run_does_not_load_model(self):
        plan_path, _ = self._write_plan([self._job('dry')])
        with mock.patch(
            'sars_adapter.run_completion._load_model',
            side_effect=AssertionError('model must not load'),
        ):
            summary = run_completion_series(
                self._config(plan_path, dry_run=True)
            )
        self.assertEqual(summary['records'][0]['status'], 'planned')

    def test_runtime_loads_model_once_and_keeps_query_selection_isolated(self):
        jobs = [
            self._job('one', offset=1.0),
            self._job('two', offset=2.0, mask='bottom'),
        ]
        plan_path, _ = self._write_plan(jobs)
        selections = []
        with mock.patch(
            'sars_adapter.run_completion._load_model', return_value=object()
        ) as load_model, mock.patch(
            'sars_adapter.run_completion.build_train_prompt_provider',
            side_effect=self._fake_prompt_provider(selections),
        ), mock.patch(
            'sars_adapter.run_completion.complete_f64_with_model',
            side_effect=self._fake_completion,
        ):
            summary = run_completion_series(self._config(plan_path))

        self.assertEqual(load_model.call_count, 1)
        self.assertEqual([row['status'] for row in summary['records']], [
            'completed', 'completed'
        ])
        self.assertEqual(len(selections), 2)
        self.assertNotEqual(selections[0], selections[1])

    def test_resume_requires_complete_provenance_binding(self):
        plan_path, _ = self._write_plan([self._job('resume')])
        selections = []
        patches = (
            mock.patch(
                'sars_adapter.run_completion._load_model', return_value=object()
            ),
            mock.patch(
                'sars_adapter.run_completion.build_train_prompt_provider',
                side_effect=self._fake_prompt_provider(selections),
            ),
            mock.patch(
                'sars_adapter.run_completion.complete_f64_with_model',
                side_effect=self._fake_completion,
            ),
        )
        with patches[0], patches[1], patches[2]:
            first = run_completion_series(self._config(plan_path))
        result_path = first['records'][0]['output_path']

        with mock.patch(
            'sars_adapter.run_completion._load_model',
            side_effect=AssertionError('valid resume must not load model'),
        ):
            resumed = run_completion_series(self._config(plan_path))
        self.assertEqual(resumed['records'][0]['status'], 'skipped_existing')

        with open(result_path, 'rb') as stream:
            result = pickle.load(stream)
        self.assertEqual(
            result['provenance']['official_upstream']['repository_url'],
            'https://github.com/fanglaosi/Skeleton-in-Context',
        )
        self.assertEqual(
            result['provenance']['adapter_fork']['repository_url'],
            'https://github.com/Hyakuri/Skeleton-in-Context',
        )
        self.assertEqual(
            result['repository_commit'],
            result['provenance']['adapter_fork']['commit'],
        )
        self.assertEqual(
            len(result['provenance']['prompt_pool']['manifest_sha256']), 64
        )
        result['provenance']['checkpoint']['sha256'] = '0' * 64
        result['result_hash'] = _package_hash(result, 'result_hash')
        with open(result_path, 'wb') as stream:
            pickle.dump(result, stream, protocol=4)

        selections = []
        with mock.patch(
            'sars_adapter.run_completion._load_model', return_value=object()
        ) as load_model, mock.patch(
            'sars_adapter.run_completion.build_train_prompt_provider',
            side_effect=self._fake_prompt_provider(selections),
        ), mock.patch(
            'sars_adapter.run_completion.complete_f64_with_model',
            side_effect=self._fake_completion,
        ):
            rerun = run_completion_series(self._config(plan_path))
        self.assertEqual(load_model.call_count, 1)
        self.assertEqual(rerun['records'][0]['status'], 'completed')

    def test_resume_rejects_nonfinite_completed_keypoint(self):
        plan_path, _ = self._write_plan([self._job('nonfinite')])
        patches = (
            mock.patch(
                'sars_adapter.run_completion._load_model', return_value=object()
            ),
            mock.patch(
                'sars_adapter.run_completion.build_train_prompt_provider',
                side_effect=self._fake_prompt_provider([]),
            ),
            mock.patch(
                'sars_adapter.run_completion.complete_f64_with_model',
                side_effect=self._fake_completion,
            ),
        )
        with patches[0], patches[1], patches[2]:
            first = run_completion_series(self._config(plan_path))
        result_path = first['records'][0]['output_path']
        with open(result_path, 'rb') as stream:
            result = pickle.load(stream)
        result['completed_keypoint'][0, 0, 0, 0] = np.nan
        result['result_hash'] = _package_hash(result, 'result_hash')
        with open(result_path, 'wb') as stream:
            pickle.dump(result, stream, protocol=4)

        with mock.patch(
            'sars_adapter.run_completion._load_model', return_value=object()
        ) as load_model, mock.patch(
            'sars_adapter.run_completion.build_train_prompt_provider',
            side_effect=self._fake_prompt_provider([]),
        ), mock.patch(
            'sars_adapter.run_completion.complete_f64_with_model',
            side_effect=self._fake_completion,
        ):
            rerun = run_completion_series(self._config(plan_path))

        self.assertEqual(load_model.call_count, 1)
        self.assertEqual(rerun['records'][0]['status'], 'completed')

    def test_continue_on_error_preserves_completed_jobs_and_progress(self):
        plan_path, _ = self._write_plan([
            self._job('success', offset=1.0),
            self._job('failure', offset=2.0, mask='bottom'),
        ])
        completions = [
            self._fake_completion(
                masked_keypoint=_query('x')['masked_keypoint'],
                missing_mask=_query('x')['missing_mask'],
                mask_policy='allow_ood_explicit',
            ),
            RuntimeError('expected failure'),
        ]
        output = io.StringIO()
        with mock.patch(
            'sars_adapter.run_completion._load_model', return_value=object()
        ), mock.patch(
            'sars_adapter.run_completion.build_train_prompt_provider',
            side_effect=self._fake_prompt_provider([]),
        ), mock.patch(
            'sars_adapter.run_completion.complete_f64_with_model',
            side_effect=completions,
        ), contextlib.redirect_stdout(output):
            summary = run_completion_series(self._config(
                plan_path, continue_on_error=True
            ))

        self.assertEqual([row['status'] for row in summary['records']], [
            'completed', 'failed'
        ])
        self.assertIn('success', output.getvalue())
        self.assertIn('failure', output.getvalue())
        self.assertIn('[1/2]', output.getvalue())
        self.assertIn('[2/2]', output.getvalue())
        with open(summary['summary_path'], 'r', encoding='utf-8') as stream:
            saved = json.load(stream)
        self.assertEqual(saved['completed_count'], 1)
        self.assertEqual(saved['failed_count'], 1)

    def test_strict_failure_saves_summary_before_raising(self):
        plan_path, _ = self._write_plan([self._job('strict-failure')])
        with mock.patch(
            'sars_adapter.run_completion._load_model', return_value=object()
        ), mock.patch(
            'sars_adapter.run_completion.build_train_prompt_provider',
            side_effect=self._fake_prompt_provider([]),
        ), mock.patch(
            'sars_adapter.run_completion.complete_f64_with_model',
            side_effect=RuntimeError('strict failure'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'strict failure'):
                run_completion_series(self._config(plan_path))
        summary_path = os.path.join(
            self.output_root, 'completion_series_summary.json'
        )
        with open(summary_path, 'r', encoding='utf-8') as stream:
            saved = json.load(stream)
        self.assertEqual(saved['records'][0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
