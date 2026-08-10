"""SiC 外部补全 public run-plan 的严格、可审计契约。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import ntpath
import os
import posixpath
import re
from collections.abc import Mapping, Sequence

from sars_adapter.contracts import load_completion_query


PLAN_FORMAT = 'sars_inter_external_completion_run_plan'
PLAN_VERSION = 1
PLAN_KEYS = {'format', 'version', 'method_profile', 'jobs', 'plan_hash'}
JOB_KEYS = {
    'job_id', 'dataset_profile', 'mask', 'model', 'source_split',
    'completion_scope', 'query_path', 'query_hash', 'result_relative_path',
    'consumer_models', 'expected_sample_count',
}
PRIVATE_FIELD_NAMES = {
    'label', 'labels', 'gt', 'gtlabel', 'groundtruth',
    'cleanreference', 'cleanreferenceid', 'cleanreferenceids',
    'cleanreferenceidentifier', 'cleanreferenceidentifiers',
    'cleankeypoint', 'cleankeypoints', 'cleanskeleton',
    'recognitionscore', 'recognitionscores', 'beforescores', 'afterscores',
    'sidecar', 'sidecarpath', 'evaluationsidecar',
    'sourcedataset', 'sourcedatasetpath', 'datasetpath',
    'triggermanifest', 'triggermanifestpath',
}


def _normalized_field_name(value):
    return ''.join(character for character in str(value).lower() if character.isalnum())


def _reject_private_fields(value, location='plan'):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError('run-plan keys must be strings at {}'.format(location))
            normalized_key = _normalized_field_name(key)
            if (
                normalized_key in PRIVATE_FIELD_NAMES
                or normalized_key.startswith('cleanreference')
                or normalized_key.startswith('cleankeypoint')
                or normalized_key.startswith('recognitionscore')
                or normalized_key.startswith('sourcedataset')
                or normalized_key.startswith('sidecar')
            ):
                raise ValueError(
                    'private field is not allowed in public run-plan {}: {}'.format(
                        location, key
                    )
                )
            _reject_private_fields(child, '{}.{}'.format(location, key))
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_private_fields(child, '{}[{}]'.format(location, index))
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('run-plan floating values must be finite')
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(
            'public run-plan must contain JSON values at {}: {}'.format(
                location, type(value)
            )
        )


def _plan_hash(plan):
    payload = {key: value for key, value in plan.items() if key != 'plan_hash'}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_result_path(value):
    raw = str(value or '').strip()
    normalized = raw.replace('\\', '/')
    drive, _ = ntpath.splitdrive(raw)
    parts = normalized.split('/')
    if (
        not raw
        or drive
        or ntpath.isabs(raw)
        or posixpath.isabs(normalized)
        or any(part in {'', '.', '..'} for part in parts)
    ):
        raise ValueError('unsafe result_relative_path: {}'.format(value))
    return '/'.join(parts)


def validate_run_plan(plan):
    """校验 public plan、query 绑定与输出路径安全性。"""
    if not isinstance(plan, dict):
        raise TypeError('external completion run-plan must be a dict')
    value = copy.deepcopy(plan)
    _reject_private_fields(value)
    if value.get('format') != PLAN_FORMAT or int(value.get('version', -1)) != PLAN_VERSION:
        raise ValueError('unsupported external completion run-plan format/version')
    missing = PLAN_KEYS - set(value)
    if missing:
        raise ValueError('run-plan is missing required fields: {}'.format(sorted(missing)))
    unsupported = set(value) - PLAN_KEYS
    if unsupported:
        raise ValueError('run-plan contains unsupported fields: {}'.format(sorted(unsupported)))
    if value.get('plan_hash') != _plan_hash(value):
        raise ValueError('run-plan plan_hash mismatch')
    jobs = value.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('run-plan jobs must be a non-empty list')
    method_profile = value.get('method_profile')
    if not isinstance(method_profile, dict):
        raise TypeError('run-plan method_profile must be a dict')
    method_name = str(method_profile.get('method_name') or '').strip()
    if not method_name:
        raise ValueError('run-plan method_profile.method_name is required')
    if method_profile.get('method_name') != method_name:
        raise ValueError('run-plan method_profile must use canonical text')
    method_profile['method_name'] = method_name
    checkpoint_sha256 = str(
        method_profile.get('checkpoint_sha256') or ''
    ).strip().lower()
    if not re.fullmatch(r'[0-9a-f]{64}', checkpoint_sha256):
        raise ValueError(
            'run-plan method_profile.checkpoint_sha256 must be a 64-character SHA256'
        )
    method_profile['checkpoint_sha256'] = checkpoint_sha256

    seen_ids = set()
    seen_results = set()
    seen_identities = set()
    normalized_jobs = []
    for index, raw_job in enumerate(jobs):
        location = 'jobs[{}]'.format(index)
        if not isinstance(raw_job, dict):
            raise TypeError('{} must be a dict'.format(location))
        missing = JOB_KEYS - set(raw_job)
        if missing:
            raise ValueError('{} is missing fields: {}'.format(location, sorted(missing)))
        unsupported = set(raw_job) - JOB_KEYS
        if unsupported:
            raise ValueError('{} contains unsupported fields: {}'.format(location, sorted(unsupported)))
        job = copy.deepcopy(raw_job)
        raw_job_id = job.get('job_id')
        job_id = str(raw_job_id or '').strip()
        if not job_id or raw_job_id != job_id:
            raise ValueError('{} job_id must be non-empty'.format(location))
        if job_id in seen_ids:
            raise ValueError('duplicate job_id: {}'.format(job_id))
        seen_ids.add(job_id)

        result_path = _safe_relative_result_path(job.get('result_relative_path'))
        if job.get('result_relative_path') != result_path:
            raise ValueError(
                '{} result_relative_path must use canonical forward slashes'.format(
                    location
                )
            )
        result_identity = result_path.casefold()
        if result_identity in seen_results:
            raise ValueError('duplicate result_relative_path: {}'.format(result_path))
        seen_results.add(result_identity)

        raw_query_path = str(job.get('query_path') or '')
        query_path = os.path.abspath(os.path.expanduser(raw_query_path))
        if raw_query_path != query_path:
            raise ValueError('{} query_path must be absolute'.format(location))
        if not os.path.isfile(query_path):
            raise FileNotFoundError(query_path)
        query = load_completion_query(query_path)
        if str(job.get('query_hash') or '') != query['query_hash']:
            raise ValueError('{} query_hash does not match query'.format(location))
        if str(job.get('dataset_profile') or '') != str(query['dataset_profile']):
            raise ValueError('{} dataset_profile does not match query'.format(location))
        if str(job.get('source_split') or '') != str(query['source_split']):
            raise ValueError('{} source_split does not match query'.format(location))
        if str(job.get('completion_scope') or '') != str(query['completion_scope']):
            raise ValueError('{} completion_scope does not match query'.format(location))
        if int(job.get('expected_sample_count', -1)) != len(query['sample_order']):
            raise ValueError('{} expected_sample_count does not match query'.format(location))

        model = job.get('model')
        normalized_model = None if model is None else str(model).strip() or None
        if model != normalized_model:
            raise ValueError('{} model must use canonical text or null'.format(location))
        model = normalized_model
        if query['completion_scope'] == 'trigger_selected' and model is None:
            raise ValueError('trigger_selected job requires model identity')
        identity = (
            str(job['dataset_profile']), str(job['mask']), model,
            str(job['source_split']), str(job['completion_scope']),
        )
        if identity in seen_identities:
            raise ValueError('duplicate job identity: {}'.format(identity))
        seen_identities.add(identity)

        raw_consumers = job.get('consumer_models')
        if not isinstance(raw_consumers, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in raw_consumers
        ):
            raise ValueError('{} consumer_models must contain strings'.format(location))
        consumers = list(raw_consumers)
        if not consumers or len(consumers) != len(set(consumers)):
            raise ValueError('{} consumer_models must be non-empty and unique'.format(location))
        if not isinstance(job.get('expected_sample_count'), int):
            raise TypeError('{} expected_sample_count must be int'.format(location))
        normalized_jobs.append(job)

    canonical_job_ids = sorted(item['job_id'] for item in normalized_jobs)
    if [item['job_id'] for item in normalized_jobs] != canonical_job_ids:
        raise ValueError('run-plan jobs must use canonical job_id order')
    value['method_profile'] = method_profile
    value['jobs'] = normalized_jobs
    return value


def build_run_plan(jobs, method_profile=None):
    """由公开 job 字段构建并立即校验一个稳定 hash 的 plan。"""
    plan = {
        'format': PLAN_FORMAT,
        'version': PLAN_VERSION,
        'method_profile': copy.deepcopy(dict(method_profile or {})),
        'jobs': sorted(
            copy.deepcopy(list(jobs)),
            key=lambda item: str(item.get('job_id') or ''),
        ),
    }
    plan['plan_hash'] = _plan_hash(plan)
    return validate_run_plan(plan)


def load_run_plan(path):
    with open(os.path.abspath(os.path.expanduser(os.fspath(path))), 'r', encoding='utf-8') as stream:
        return validate_run_plan(json.load(stream))


__all__ = [
    'PLAN_FORMAT', 'PLAN_VERSION', 'build_run_plan', 'load_run_plan',
    'validate_run_plan',
]
