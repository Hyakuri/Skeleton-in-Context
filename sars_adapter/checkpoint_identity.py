"""SiC checkpoint 的可移植身份清单与跨设备校验。"""

from __future__ import annotations

import copy
import hashlib
import json
import ntpath
import os
import posixpath
import re
import shutil
import tempfile

import torch


MANIFEST_FORMAT = "skeleton_in_context_checkpoint_identity_manifest"
MANIFEST_VERSION = 1
MANIFEST_FILENAME = "checkpoint_identity.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRAINING_IDENTITY_FIELDS = {
    "format",
    "version",
    "tasks",
    "task_scope",
    "subset_seed",
    "training_seed",
    "effective_config_sha256",
    "subset_manifest_sha256",
}
_SUPPORTED_TASKS = {"PE", "MP", "MC", "FPE"}


def _absolute_path(value):
    raw = os.path.expanduser(os.fspath(value))
    if os.name == "nt" and raw.startswith("\\\\?\\"):
        return raw
    return os.path.abspath(raw)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(_absolute_path(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload):
    value = {
        key: child for key, child in payload.items() if key != "manifest_hash"
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_filename(value, field_name):
    raw = str(value or "").strip().replace("\\", "/")
    drive, _ = ntpath.splitdrive(raw)
    parts = raw.split("/")
    if (
        not raw
        or drive
        or ntpath.isabs(raw)
        or posixpath.isabs(raw)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("{} must be a safe relative filename".format(field_name))
    normalized = posixpath.normpath(raw)
    if normalized != raw or normalized.startswith("../"):
        raise ValueError("{} must be a safe relative filename".format(field_name))
    return normalized


def _validate_file_record(record, field_name):
    if not isinstance(record, dict):
        raise TypeError("{} must be a dict".format(field_name))
    if set(record) != {"filename", "sha256", "size_bytes"}:
        raise ValueError("{} has unsupported or missing fields".format(field_name))
    filename = _safe_relative_filename(record.get("filename"), field_name)
    sha256 = str(record.get("sha256") or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError("{}.sha256 must be a 64-character SHA256".format(field_name))
    size_bytes = record.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError("{}.size_bytes must be a positive int".format(field_name))
    return {
        "filename": filename,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _validate_training_identity(identity, require_mc_only=True):
    if not isinstance(identity, dict):
        raise TypeError("training_identity must be a dict")
    if set(identity) != _TRAINING_IDENTITY_FIELDS:
        raise ValueError(
            "training identity has unsupported or missing fields"
        )
    if (
        identity.get("format") != "skeleton_in_context_training_identity"
        or int(identity.get("version", -1)) != 1
    ):
        raise ValueError("unsupported training identity format/version")
    tasks = identity.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(task, str) for task in tasks)
        or len(set(tasks)) != len(tasks)
        or not set(tasks).issubset(_SUPPORTED_TASKS)
    ):
        raise ValueError("training identity tasks are invalid")
    expected_scope = "single_task" if len(tasks) == 1 else "multi_task"
    if identity.get("task_scope") != expected_scope:
        raise ValueError("training identity task_scope mismatch")
    for name in ("subset_seed", "training_seed"):
        value = identity.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("training identity {} must be an int".format(name))
    normalized = copy.deepcopy(identity)
    for name in ("effective_config_sha256", "subset_manifest_sha256"):
        value = str(identity.get(name) or "").strip().lower()
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(
                "training identity {} must be a SHA256".format(name)
            )
        normalized[name] = value
    if require_mc_only and (
        normalized["tasks"] != ["MC"]
        or normalized["task_scope"] != "single_task"
    ):
        raise ValueError("formal completion requires an MC-only checkpoint")
    return normalized


def validate_completion_checkpoint_identity(
    checkpoint, source_config, policy="require_mc_only"
):
    """校验 checkpoint 与训练配置、任务范围是否一致。"""
    policy = str(policy or "require_mc_only")
    if policy not in {"require_mc_only", "allow_legacy"}:
        raise ValueError("unsupported checkpoint identity policy")
    identity = checkpoint.get("training_identity")
    if identity is None:
        if policy == "require_mc_only":
            raise ValueError("formal completion checkpoint is missing training identity")
        return None
    if not isinstance(identity, dict):
        raise ValueError("checkpoint training identity must be a dict")
    identity = _validate_training_identity(
        identity, require_mc_only=(policy == "require_mc_only")
    )
    expected_config_hash = str(
        identity.get("effective_config_sha256") or ""
    ).lower()
    actual_config_hash = sha256_file(source_config).lower()
    if expected_config_hash != actual_config_hash:
        raise ValueError("checkpoint source config hash does not match source_config")
    return copy.deepcopy(identity)


def validate_checkpoint_identity_manifest(payload):
    if not isinstance(payload, dict):
        raise TypeError("checkpoint identity manifest must be a dict")
    value = copy.deepcopy(payload)
    required = {
        "format",
        "version",
        "method_name",
        "checkpoint_identity_policy",
        "checkpoint",
        "source_config",
        "training_identity",
        "manifest_hash",
    }
    if set(value) != required:
        raise ValueError("checkpoint identity manifest has unsupported or missing fields")
    if (
        value.get("format") != MANIFEST_FORMAT
        or int(value.get("version", -1)) != MANIFEST_VERSION
    ):
        raise ValueError("unsupported checkpoint identity manifest format/version")
    if value.get("method_name") != "skeleton_in_context":
        raise ValueError("checkpoint identity manifest method_name mismatch")
    policy = str(value.get("checkpoint_identity_policy") or "")
    if policy != "require_mc_only":
        raise ValueError(
            "identity manifest checkpoint_identity_policy must be require_mc_only"
        )
    value["checkpoint_identity_policy"] = policy
    value["checkpoint"] = _validate_file_record(
        value.get("checkpoint"), "checkpoint"
    )
    value["source_config"] = _validate_file_record(
        value.get("source_config"), "source_config"
    )
    value["training_identity"] = _validate_training_identity(
        value.get("training_identity"), require_mc_only=True
    )
    manifest_hash = str(value.get("manifest_hash") or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(manifest_hash):
        raise ValueError("manifest_hash must be a 64-character SHA256")
    if manifest_hash != _canonical_hash(value):
        raise ValueError("checkpoint identity manifest_hash mismatch")
    value["manifest_hash"] = manifest_hash
    return value


def _artifact_path(manifest_path, record):
    return _absolute_path(
        os.path.join(
            os.path.dirname(_absolute_path(manifest_path)),
            *record["filename"].split("/"),
        )
    )


def load_checkpoint_identity_manifest(path, verify_files=True):
    manifest_path = _absolute_path(path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as stream:
        value = json.load(stream)
    # 先验证路径字段，避免绝对路径被 manifest_hash 错误掩盖。
    if isinstance(value, dict):
        for name in ("checkpoint", "source_config"):
            record = value.get(name)
            if isinstance(record, dict):
                _safe_relative_filename(record.get("filename"), name)
    manifest = validate_checkpoint_identity_manifest(value)
    if verify_files:
        for name in ("checkpoint", "source_config"):
            record = manifest[name]
            artifact_path = _artifact_path(manifest_path, record)
            if not os.path.isfile(artifact_path):
                raise FileNotFoundError(artifact_path)
            if os.path.getsize(artifact_path) != record["size_bytes"]:
                raise ValueError("{} size does not match identity manifest".format(name))
            if sha256_file(artifact_path) != record["sha256"]:
                raise ValueError("{} SHA256 does not match identity manifest".format(name))
    return manifest


def _is_configured_path(value):
    text = str(value or "").strip()
    return bool(text and "<" not in text and ">" not in text)


def resolve_checkpoint_source(config):
    resolved = copy.deepcopy(dict(config or {}))
    configured_mode = resolved.get("checkpoint_source_mode")
    if configured_mode is None:
        configured_mode = (
            "identity_manifest"
            if _is_configured_path(resolved.get("checkpoint_identity_manifest_path"))
            else "direct_path"
        )
    mode = str(configured_mode)
    if mode not in {"identity_manifest", "direct_path"}:
        raise ValueError("unsupported checkpoint_source_mode")
    if mode == "identity_manifest":
        manifest_path = resolved.get("checkpoint_identity_manifest_path")
        if not _is_configured_path(manifest_path):
            raise ValueError("checkpoint_identity_manifest_path is required")
        manifest_path = _absolute_path(manifest_path)
        manifest = load_checkpoint_identity_manifest(manifest_path, verify_files=True)
        configured_policy = str(
            resolved.get("checkpoint_identity_policy") or "require_mc_only"
        )
        if configured_policy != "require_mc_only":
            raise ValueError(
                "identity_manifest mode requires checkpoint_identity_policy="
                "require_mc_only"
            )
        configured_sha256 = str(
            resolved.get("checkpoint_sha256") or ""
        ).strip().lower()
        if (
            configured_sha256
            and configured_sha256 != manifest["checkpoint"]["sha256"]
        ):
            raise ValueError(
                "checkpoint_sha256 conflicts with checkpoint identity manifest"
            )
        checkpoint_path = _artifact_path(manifest_path, manifest["checkpoint"])
        source_config = _artifact_path(manifest_path, manifest["source_config"])
        for name, actual in (
            ("checkpoint_path", checkpoint_path),
            ("source_config", source_config),
        ):
            configured = resolved.get(name)
            if _is_configured_path(configured) and _absolute_path(configured) != actual:
                raise ValueError("{} conflicts with checkpoint identity manifest".format(name))
        resolved.update(
            {
                "checkpoint_source_mode": mode,
                "checkpoint_identity_manifest_path": manifest_path,
                "checkpoint_identity_manifest_sha256": manifest["manifest_hash"],
                "checkpoint_path": checkpoint_path,
                "source_config": source_config,
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "checkpoint_identity_policy": manifest[
                    "checkpoint_identity_policy"
                ],
                "checkpoint_training_identity": copy.deepcopy(
                    manifest.get("training_identity")
                ),
            }
        )
        return resolved

    for name in ("checkpoint_path", "source_config"):
        if not _is_configured_path(resolved.get(name)):
            raise ValueError("{} is required for direct_path mode".format(name))
        resolved[name] = _absolute_path(resolved[name])
        if not os.path.isfile(resolved[name]):
            raise FileNotFoundError(resolved[name])
    actual_sha256 = sha256_file(resolved["checkpoint_path"])
    expected_sha256 = str(resolved.get("checkpoint_sha256") or "").strip().lower()
    if expected_sha256 and expected_sha256 != actual_sha256:
        raise ValueError("direct checkpoint SHA256 does not match configured value")
    resolved.update(
        {
            "checkpoint_source_mode": mode,
            "checkpoint_identity_manifest_path": None,
            "checkpoint_identity_manifest_sha256": None,
            "checkpoint_sha256": actual_sha256,
        }
    )
    return resolved


def _atomic_json_dump(path, payload):
    path = _absolute_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tmp-manifest-", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _build_manifest(
    *,
    checkpoint_name,
    checkpoint_hash,
    checkpoint_size,
    config_name,
    config_hash,
    config_size,
    training_identity,
):
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "method_name": "skeleton_in_context",
        "checkpoint_identity_policy": "require_mc_only",
        "checkpoint": {
            "filename": checkpoint_name,
            "sha256": checkpoint_hash,
            "size_bytes": checkpoint_size,
        },
        "source_config": {
            "filename": config_name,
            "sha256": config_hash,
            "size_bytes": config_size,
        },
        "training_identity": copy.deepcopy(training_identity),
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    return validate_checkpoint_identity_manifest(manifest)


def freeze_checkpoint_bundle(config):
    value = copy.deepcopy(dict(config or {}))
    raw_output_dir = str(value.get("output_dir") or "").strip()
    if not raw_output_dir or "<" in raw_output_dir or ">" in raw_output_dir:
        raise ValueError("output_dir is required")
    output_dir = _absolute_path(raw_output_dir)
    direct = resolve_checkpoint_source(
        {
            **value,
            "checkpoint_source_mode": "direct_path",
        }
    )
    policy = str(value.get("checkpoint_identity_policy") or "require_mc_only")
    if policy != "require_mc_only":
        raise ValueError(
            "formal checkpoint bundle policy must be require_mc_only; "
            "use direct_path for legacy smoke inference"
        )
    checkpoint_name = os.path.basename(direct["checkpoint_path"])
    config_name = os.path.basename(direct["source_config"])
    if checkpoint_name.casefold() == config_name.casefold():
        config_name = "effective_config_{}".format(config_name)
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    if bool(value.get("dry_run", False)):
        checkpoint = torch.load(direct["checkpoint_path"], map_location="cpu")
        training_identity = validate_completion_checkpoint_identity(
            checkpoint, direct["source_config"], policy=policy
        )
        config_hash = sha256_file(direct["source_config"])
        manifest = _build_manifest(
            checkpoint_name=checkpoint_name,
            checkpoint_hash=direct["checkpoint_sha256"],
            checkpoint_size=os.path.getsize(direct["checkpoint_path"]),
            config_name=config_name,
            config_hash=config_hash,
            config_size=os.path.getsize(direct["source_config"]),
            training_identity=training_identity,
        )
        return {
            "status": "planned",
            "output_dir": output_dir,
            "manifest_path": manifest_path,
            "checkpoint_sha256": direct["checkpoint_sha256"],
            "manifest_hash": manifest["manifest_hash"],
        }

    parent_dir = os.path.dirname(output_dir)
    os.makedirs(parent_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=".tmp-sic-checkpoint-", dir=parent_dir)
    published = False
    try:
        staged_checkpoint = os.path.join(staging_dir, checkpoint_name)
        staged_config = os.path.join(staging_dir, config_name)
        shutil.copy2(direct["checkpoint_path"], staged_checkpoint)
        shutil.copy2(direct["source_config"], staged_config)
        staged_checkpoint_hash = sha256_file(staged_checkpoint)
        if staged_checkpoint_hash != direct["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed while creating bundle")
        checkpoint = torch.load(staged_checkpoint, map_location="cpu")
        training_identity = validate_completion_checkpoint_identity(
            checkpoint, staged_config, policy=policy
        )
        manifest = _build_manifest(
            checkpoint_name=checkpoint_name,
            checkpoint_hash=staged_checkpoint_hash,
            checkpoint_size=os.path.getsize(staged_checkpoint),
            config_name=config_name,
            config_hash=sha256_file(staged_config),
            config_size=os.path.getsize(staged_config),
            training_identity=training_identity,
        )
        staged_manifest = os.path.join(staging_dir, MANIFEST_FILENAME)
        _atomic_json_dump(staged_manifest, manifest)
        load_checkpoint_identity_manifest(staged_manifest, verify_files=True)

        if os.path.exists(output_dir):
            try:
                existing = load_checkpoint_identity_manifest(
                    manifest_path, verify_files=True
                )
            except Exception as error:
                raise FileExistsError(
                    "output_dir already exists and is not the same verified bundle: "
                    "{}".format(output_dir)
                ) from error
            if existing != manifest:
                raise FileExistsError(
                    "refusing to replace an existing checkpoint bundle in place: "
                    "{}".format(output_dir)
                )
            status = "reused_existing"
        else:
            os.replace(staging_dir, output_dir)
            published = True
            status = "completed"
    finally:
        if not published and os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)

    load_checkpoint_identity_manifest(manifest_path, verify_files=True)
    return {
        "status": status,
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "manifest_hash": manifest["manifest_hash"],
    }


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "freeze_checkpoint_bundle",
    "load_checkpoint_identity_manifest",
    "resolve_checkpoint_source",
    "sha256_file",
    "validate_checkpoint_identity_manifest",
    "validate_completion_checkpoint_identity",
]
