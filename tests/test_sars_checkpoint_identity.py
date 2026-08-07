import hashlib
import json
import os
import pickle
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


class CheckpointIdentityTest(unittest.TestCase):
    def _write_training_artifacts(self, root, *, tasks=None, extra_identity=None):
        tasks = list(tasks or ["MC"])
        config_path = os.path.join(root, "effective_config.yaml")
        with open(config_path, "wb") as stream:
            stream.write(("tasks: [{}]\n".format(", ".join(tasks))).encode("utf-8"))
        with open(config_path, "rb") as stream:
            config_hash = hashlib.sha256(stream.read()).hexdigest()
        checkpoint_path = os.path.join(root, "epoch_20.bin")
        training_identity = {
            "format": "skeleton_in_context_training_identity",
            "version": 1,
            "tasks": tasks,
            "task_scope": "single_task" if tasks == ["MC"] else "multi_task",
            "subset_seed": 42,
            "training_seed": 42,
            "effective_config_sha256": config_hash,
            "subset_manifest_sha256": "b" * 64,
        }
        training_identity.update(dict(extra_identity or {}))
        torch.save(
            {
                "epoch": 20,
                "model_pos": {},
                "training_identity": training_identity,
            },
            checkpoint_path,
        )
        return checkpoint_path, config_path

    @staticmethod
    def _write_query(path):
        from sars_adapter.contracts import _query_hash

        masked = np.zeros((1, 64, 17, 3), dtype=np.float32)
        missing = np.zeros((1, 64, 17), dtype=bool)
        missing[:, 16:48, 8:] = True
        query = {
            "format": "sars_inter_external_completion_query",
            "version": 2,
            "dataset_profile": "custom_worksite",
            "source_split": "test",
            "completion_scope": "all_split",
            "selection_manifest_hash": None,
            "sample_order": ["sample-a"],
            "frame_count": 64,
            "joint_count": 17,
            "missing_mask_semantics": "True=missing",
            "coordinate_contract": {
                "skeleton": "H36M17",
                "joint_order": "h36m17_sars_inter_project_order",
                "axis_order": ["x_lateral", "y_depth", "z_height"],
                "unit": "project_native",
            },
            "temporal_metadata": {"target_frame_count": 64},
            "masked_keypoint": masked,
            "missing_mask": missing,
            "metadata": {"mask": "upper"},
        }
        query["query_hash"] = _query_hash(query)
        with open(path, "wb") as stream:
            pickle.dump(query, stream, protocol=4)

    def test_direct_path_mode_computes_checkpoint_identity(self):
        from sars_adapter.checkpoint_identity import resolve_checkpoint_source

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(root)
            resolved = resolve_checkpoint_source(
                {
                    "checkpoint_source_mode": "direct_path",
                    "checkpoint_path": checkpoint_path,
                    "source_config": config_path,
                }
            )

        self.assertEqual(resolved["checkpoint_source_mode"], "direct_path")
        self.assertEqual(len(resolved["checkpoint_sha256"]), 64)
        self.assertIsNone(resolved["checkpoint_identity_manifest_path"])

    def test_completion_direct_configs_default_to_formal_manifest_mode(self):
        from sars_adapter.run_completion import build_direct_run_config
        from sars_adapter.run_completion_series import (
            build_direct_run_config as build_series_direct_run_config,
        )

        single = build_direct_run_config()
        series = build_series_direct_run_config()

        self.assertEqual(single["checkpoint_source_mode"], "identity_manifest")
        self.assertEqual(series["checkpoint_source_mode"], "identity_manifest")
        self.assertIn("checkpoint_identity_manifest_path", single)
        self.assertIn("checkpoint_identity_manifest_path", series)

    def test_completion_runtime_resolves_formal_manifest(self):
        from sars_adapter.checkpoint_identity import freeze_checkpoint_bundle
        from sars_adapter.run_completion import _resolve_runtime_paths

        with tempfile.TemporaryDirectory() as root:
            source_root = os.path.join(root, "source")
            bundle_root = os.path.join(root, "bundle")
            data_root = os.path.join(root, "data")
            os.makedirs(source_root)
            os.makedirs(data_root)
            checkpoint_path, config_path = self._write_training_artifacts(source_root)
            summary = freeze_checkpoint_bundle(
                {
                    "checkpoint_path": checkpoint_path,
                    "source_config": config_path,
                    "output_dir": bundle_root,
                }
            )
            resolved = _resolve_runtime_paths(
                {
                    "checkpoint_source_mode": "identity_manifest",
                    "checkpoint_identity_manifest_path": summary["manifest_path"],
                    "data_root": data_root,
                    "coordinate_transform_mode": "project_h36m17_prompt_aligned_v1",
                    "demonstration_selection_policy": "per_sample_fixed",
                }
            )

        self.assertEqual(resolved["checkpoint_sha256"], summary["checkpoint_sha256"])
        self.assertEqual(
            resolved["checkpoint_identity_manifest_sha256"],
            summary["manifest_hash"],
        )
        self.assertTrue(resolved["checkpoint_path"].endswith("epoch_20.bin"))

    def test_formal_manifest_rejects_conflicting_configured_checkpoint_hash(self):
        from sars_adapter.checkpoint_identity import (
            freeze_checkpoint_bundle,
            resolve_checkpoint_source,
        )

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(root)
            summary = freeze_checkpoint_bundle(
                {
                    "checkpoint_path": checkpoint_path,
                    "source_config": config_path,
                    "output_dir": os.path.join(root, "bundle"),
                }
            )
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256 conflicts"):
                resolve_checkpoint_source(
                    {
                        "checkpoint_source_mode": "identity_manifest",
                        "checkpoint_identity_manifest_path": summary[
                            "manifest_path"
                        ],
                        "checkpoint_sha256": "0" * 64,
                    }
                )

    def test_formal_manifest_rejects_policy_downgrade(self):
        from sars_adapter.checkpoint_identity import freeze_checkpoint_bundle

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(root)
            with self.assertRaisesRegex(ValueError, "require_mc_only"):
                freeze_checkpoint_bundle(
                    {
                        "checkpoint_path": checkpoint_path,
                        "source_config": config_path,
                        "output_dir": os.path.join(root, "bundle"),
                        "checkpoint_identity_policy": "allow_legacy",
                    }
                )

    def test_completion_dry_run_resolves_and_reports_checkpoint_hash(self):
        from sars_adapter.run_completion import run_completion

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(root)
            query_path = os.path.join(root, "query.pkl")
            data_root = os.path.join(root, "data")
            os.makedirs(data_root)
            self._write_query(query_path)
            result = run_completion(
                {
                    "dry_run": True,
                    "query_path": query_path,
                    "output_path": os.path.join(root, "result.pkl"),
                    "checkpoint_source_mode": "direct_path",
                    "checkpoint_path": checkpoint_path,
                    "source_config": config_path,
                    "checkpoint_identity_policy": "require_mc_only",
                    "data_root": data_root,
                    "coordinate_transform_mode": (
                        "project_h36m17_prompt_aligned_v1"
                    ),
                    "demonstration_selection_policy": "per_sample_fixed",
                    "require_clean_repository": False,
                }
            )

        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["checkpoint_source_mode"], "direct_path")
        self.assertEqual(len(result["checkpoint_sha256"]), 64)

    def test_shared_runtime_reuses_resolved_checkpoint_identity(self):
        import sars_adapter.run_completion as completion

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(root)
            data_root = os.path.join(root, "data")
            os.makedirs(data_root)
            runtime = {}
            config = {
                "checkpoint_source_mode": "direct_path",
                "checkpoint_path": checkpoint_path,
                "source_config": config_path,
                "checkpoint_identity_policy": "allow_legacy",
                "data_root": data_root,
                "device": "cpu",
                "coordinate_transform_mode": "project_h36m17_prompt_aligned_v1",
                "demonstration_selection_policy": "per_sample_fixed",
                "require_clean_repository": False,
            }
            with open(config_path, "rb") as stream:
                source_config_sha256 = hashlib.sha256(stream.read()).hexdigest()
            prompt_pool = {
                "prompt_pool_manifest_sha256": "d" * 64,
                "prompt_pool_file_count": 1,
                "source_config_sha256": source_config_sha256,
            }
            with mock.patch.object(
                completion,
                "build_train_prompt_pool",
                return_value=prompt_pool,
            ), mock.patch.object(
                completion,
                "_repository_identity",
                return_value=("a" * 40, False, "b" * 64),
            ), mock.patch.object(
                completion,
                "resolve_checkpoint_source",
                wraps=completion.resolve_checkpoint_source,
            ) as resolver:
                completion.prepare_completion_runtime(
                    config, runtime=runtime, load_model=False
                )
                completion.prepare_completion_runtime(
                    config, runtime=runtime, load_model=False
                )

        self.assertEqual(resolver.call_count, 1)

    def test_freeze_bundle_is_portable_across_directories(self):
        from sars_adapter.checkpoint_identity import (
            freeze_checkpoint_bundle,
            load_checkpoint_identity_manifest,
        )

        with tempfile.TemporaryDirectory() as root:
            source_root = os.path.join(root, "source")
            bundle_root = os.path.join(root, "bundle")
            moved_root = os.path.join(root, "moved")
            os.makedirs(source_root)
            checkpoint_path, config_path = self._write_training_artifacts(source_root)
            summary = freeze_checkpoint_bundle(
                {
                    "checkpoint_path": checkpoint_path,
                    "source_config": config_path,
                    "output_dir": bundle_root,
                    "checkpoint_identity_policy": "require_mc_only",
                    "overwrite": False,
                }
            )
            shutil.copytree(bundle_root, moved_root)
            moved_manifest = os.path.join(moved_root, "checkpoint_identity.json")
            identity = load_checkpoint_identity_manifest(
                moved_manifest, verify_files=True
            )
            with open(moved_manifest, "r", encoding="utf-8") as stream:
                raw_text = stream.read()

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(identity["training_identity"]["tasks"], ["MC"])
        self.assertNotIn(source_root, raw_text)
        self.assertFalse(os.path.isabs(identity["checkpoint"]["filename"]))
        self.assertFalse(os.path.isabs(identity["source_config"]["filename"]))

    def test_freeze_requires_explicit_output_dir(self):
        from sars_adapter.checkpoint_identity import freeze_checkpoint_bundle

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(root)
            with self.assertRaisesRegex(ValueError, "output_dir is required"):
                freeze_checkpoint_bundle(
                    {
                        "dry_run": True,
                        "checkpoint_path": checkpoint_path,
                        "source_config": config_path,
                    }
                )

    def test_freeze_rejects_unknown_training_identity_fields(self):
        from sars_adapter.checkpoint_identity import freeze_checkpoint_bundle

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(
                root,
                extra_identity={"local_checkpoint_path": "C:/private/model.bin"},
            )
            with self.assertRaisesRegex(ValueError, "training identity.*fields"):
                freeze_checkpoint_bundle(
                    {
                        "checkpoint_path": checkpoint_path,
                        "source_config": config_path,
                        "output_dir": os.path.join(root, "bundle"),
                    }
                )

    def test_freeze_captures_one_consistent_config_snapshot(self):
        import sars_adapter.checkpoint_identity as checkpoint_identity

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(root)
            original_validator = (
                checkpoint_identity.validate_completion_checkpoint_identity
            )

            def mutate_source_after_validation(checkpoint, source_config, policy):
                identity = original_validator(checkpoint, source_config, policy)
                if os.path.abspath(source_config) == os.path.abspath(config_path):
                    with open(config_path, "ab") as stream:
                        stream.write(b"# changed after validation\n")
                return identity

            with mock.patch.object(
                checkpoint_identity,
                "validate_completion_checkpoint_identity",
                side_effect=mutate_source_after_validation,
            ):
                summary = checkpoint_identity.freeze_checkpoint_bundle(
                    {
                        "checkpoint_path": checkpoint_path,
                        "source_config": config_path,
                        "output_dir": os.path.join(root, "bundle"),
                    }
                )
            manifest = checkpoint_identity.load_checkpoint_identity_manifest(
                summary["manifest_path"], verify_files=True
            )

        self.assertEqual(
            manifest["source_config"]["sha256"],
            manifest["training_identity"]["effective_config_sha256"],
        )

    def test_existing_bundle_cannot_be_replaced_in_place(self):
        from sars_adapter.checkpoint_identity import (
            freeze_checkpoint_bundle,
            load_checkpoint_identity_manifest,
        )

        with tempfile.TemporaryDirectory() as root:
            first_root = os.path.join(root, "first")
            second_root = os.path.join(root, "second")
            bundle_root = os.path.join(root, "bundle")
            os.makedirs(first_root)
            os.makedirs(second_root)
            first_checkpoint, first_config = self._write_training_artifacts(first_root)
            second_checkpoint, second_config = self._write_training_artifacts(
                second_root, extra_identity={"training_seed": 7}
            )
            first = freeze_checkpoint_bundle(
                {
                    "checkpoint_path": first_checkpoint,
                    "source_config": first_config,
                    "output_dir": bundle_root,
                }
            )
            with self.assertRaises(FileExistsError):
                freeze_checkpoint_bundle(
                    {
                        "checkpoint_path": second_checkpoint,
                        "source_config": second_config,
                        "output_dir": bundle_root,
                        "overwrite": True,
                    }
                )
            unchanged = load_checkpoint_identity_manifest(
                first["manifest_path"], verify_files=True
            )

        self.assertEqual(unchanged["manifest_hash"], first["manifest_hash"])

    def test_manifest_rejects_tampered_checkpoint(self):
        from sars_adapter.checkpoint_identity import (
            freeze_checkpoint_bundle,
            load_checkpoint_identity_manifest,
        )

        with tempfile.TemporaryDirectory() as root:
            source_root = os.path.join(root, "source")
            bundle_root = os.path.join(root, "bundle")
            os.makedirs(source_root)
            checkpoint_path, config_path = self._write_training_artifacts(source_root)
            freeze_checkpoint_bundle(
                {
                    "checkpoint_path": checkpoint_path,
                    "source_config": config_path,
                    "output_dir": bundle_root,
                }
            )
            with open(os.path.join(bundle_root, "epoch_20.bin"), "ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint (size|SHA256)"):
                load_checkpoint_identity_manifest(
                    os.path.join(bundle_root, "checkpoint_identity.json"),
                    verify_files=True,
                )

    def test_runtime_rejects_manifest_identity_change(self):
        import sars_adapter.checkpoint_identity as checkpoint_identity
        from sars_adapter.run_completion import verify_completion_runtime_assets

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(root)
            data_root = os.path.join(root, "data")
            os.makedirs(data_root)
            summary = checkpoint_identity.freeze_checkpoint_bundle(
                {
                    "checkpoint_path": checkpoint_path,
                    "source_config": config_path,
                    "output_dir": os.path.join(root, "bundle"),
                }
            )
            manifest_path = summary["manifest_path"]
            with open(manifest_path, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            manifest["training_identity"]["training_seed"] = 7
            manifest["manifest_hash"] = checkpoint_identity._canonical_hash(
                manifest
            )
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            config = {
                "checkpoint_source_mode": "identity_manifest",
                "checkpoint_identity_manifest_path": manifest_path,
                "checkpoint_identity_policy": "require_mc_only",
                "data_root": data_root,
                "coordinate_transform_mode": (
                    "project_h36m17_prompt_aligned_v1"
                ),
                "demonstration_selection_policy": "per_sample_fixed",
            }
            runtime = {
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "provenance": {
                    "checkpoint": {
                        "identity_manifest_sha256": summary["manifest_hash"]
                    }
                },
            }
            with self.assertRaisesRegex(RuntimeError, "manifest changed"):
                verify_completion_runtime_assets(config, runtime)

    def test_manifest_rejects_absolute_artifact_filename(self):
        from sars_adapter.checkpoint_identity import (
            freeze_checkpoint_bundle,
            load_checkpoint_identity_manifest,
        )

        with tempfile.TemporaryDirectory() as root:
            source_root = os.path.join(root, "source")
            bundle_root = os.path.join(root, "bundle")
            os.makedirs(source_root)
            checkpoint_path, config_path = self._write_training_artifacts(source_root)
            freeze_checkpoint_bundle(
                {
                    "checkpoint_path": checkpoint_path,
                    "source_config": config_path,
                    "output_dir": bundle_root,
                }
            )
            manifest_path = os.path.join(bundle_root, "checkpoint_identity.json")
            with open(manifest_path, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            manifest["checkpoint"]["filename"] = checkpoint_path
            manifest["manifest_hash"] = "0" * 64
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with self.assertRaisesRegex(ValueError, "relative filename"):
                load_checkpoint_identity_manifest(manifest_path, verify_files=False)

    def test_freeze_rejects_non_mc_checkpoint(self):
        from sars_adapter.checkpoint_identity import freeze_checkpoint_bundle

        with tempfile.TemporaryDirectory() as root:
            checkpoint_path, config_path = self._write_training_artifacts(
                root, tasks=["PE", "MP", "MC", "FPE"]
            )
            with self.assertRaisesRegex(ValueError, "MC-only"):
                freeze_checkpoint_bundle(
                    {
                        "checkpoint_path": checkpoint_path,
                        "source_config": config_path,
                        "output_dir": os.path.join(root, "bundle"),
                    }
                )


if __name__ == "__main__":
    unittest.main()
