"""按 public run-plan 批量执行可恢复的 SiC 外部补全。"""

from __future__ import annotations

import copy
import json
import os
import pickle
import tempfile
import time
import os.path as osp
import os, sys
import numpy as np

ROOT_DIR = osp.join(osp.dirname(osp.abspath(__file__)), os.path.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sars_adapter.contracts import load_completion_query
from sars_adapter.run_completion import (
    _package_hash,
    _resolve_runtime_paths,
    prepare_completion_runtime,
    run_completion,
    verify_completion_runtime_assets,
)
from sars_adapter.series_contracts import load_run_plan


def build_direct_run_config():
    """集中配置 public plan 批量执行入口。"""
    
    SARS_INTER_1stRecogResults_dirname = "SiC_MC_CustomDataset_20260810141623"
    
    return {
        'checkpoint_source_mode': 'identity_manifest',  # 正式使用身份清单；smoke 可改为 direct_path。
        'checkpoint_identity_manifest_path': r"K:\ExternalCompletionBaselines\Skeleton-in-Context\checkpoints\sic_mc_20260806232122\frozen_checkpoints\checkpoint_identity.json",  # 正式 bundle 中的身份清单。
        'plan_path': rf"\\?\K:\SARS-Inter_DL\PP_Informatics_MajorRevision\{SARS_INTER_1stRecogResults_dirname}\custom__sic_custom_all_split\external_completion_export\public_query\external_completion_run_plan.json",  # 仅含公开 query job 的 V1 plan。
        'output_root': rf"\\?\K:\SARS-Inter_DL\PP_Informatics_MajorRevision\{SARS_INTER_1stRecogResults_dirname}\sic_completion_results",  # 所有相对结果路径的安全根目录。
        'checkpoint_path': None,  # 仅 direct_path 冒烟模式填写；程序自动计算 SHA256。
        'source_config': None,  # 仅 direct_path 填写，且必须与 checkpoint 来自同一次训练。
        'checkpoint_identity_policy': 'require_mc_only',  # identity_manifest 固定 require_mc_only；allow_legacy 只允许 direct_path 旧权重冒烟。
        'data_root': r"K:\ExternalCompletionBaselines\Skeleton-in-Context\data",  # 只读取 3DPW_MC/train demonstrations。
        'device': 'cuda:0',  # 可填写 cuda:0 或 cpu。
        'dry_run': False,  # True=校验 plan 与 checkpoint 身份并记录 planned，不加载模型。
        'resume': True,  # True=仅跳过通过完整绑定校验的已有结果。
        'strict': True,  # True=失败且未启用 continue_on_error 时抛错。
        'continue_on_error': False,  # True=记录失败并继续后续 job。
        'mask_policy': 'allow_ood_explicit',  # 显式接受项目长时遮挡 mask。
        'demonstration_seed': 42,  # train demonstration 的稳定选择种子。
        'demonstration_selection_policy': 'per_sample_fixed',  # 每样本固定 demonstration。
        'coordinate_transform_mode': 'project_h36m17_prompt_aligned_v1',  # 正式坐标适配。
    }


def _absolute_path(value):
    raw = os.path.expanduser(os.fspath(value))
    if os.name == 'nt' and raw.startswith('\\\\?\\'):
        return raw
    return os.path.abspath(raw)


def _atomic_json_dump(path, payload):
    output_path = _absolute_path(path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix='.tmp-', suffix='.json', dir=output_dir or None
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _load_pickle(path):
    with open(_absolute_path(path), 'rb') as stream:
        return pickle.load(stream)


def _validate_existing_result(path, query, expected_provenance):
    result = _load_pickle(path)
    if not isinstance(result, dict):
        raise TypeError('existing completion result must be a dict')
    if (
        result.get('format') != 'sars_inter_external_completion_result'
        or int(result.get('version', -1)) != 2
    ):
        raise ValueError('existing completion result format/version mismatch')
    if result.get('result_hash') != _package_hash(result, 'result_hash'):
        raise ValueError('existing completion result_hash mismatch')
    if result.get('query_hash') != query['query_hash']:
        raise ValueError('existing completion query_hash mismatch')
    if list(result.get('sample_order') or []) != list(query['sample_order']):
        raise ValueError('existing completion sample_order mismatch')
    if result.get('checkpoint_identity') != expected_provenance['checkpoint']['sha256']:
        raise ValueError('existing completion checkpoint identity mismatch')
    result_provenance = result.get('provenance') or {}
    for section_name in ('checkpoint', 'prompt_pool', 'completion_policy'):
        if result_provenance.get(section_name) != expected_provenance.get(
            section_name
        ):
            raise ValueError(
                'existing completion stable provenance mismatch: {}'.format(
                    section_name
                )
            )
    if result.get('method_name') != 'skeleton_in_context':
        raise ValueError('existing completion method identity mismatch')
    for field_name in ('dataset_profile', 'source_split', 'coordinate_contract'):
        if result.get(field_name) != query.get(field_name):
            raise ValueError(
                'existing completion {} mismatch'.format(field_name)
            )
    if int(result.get('frame_count', -1)) != 64:
        raise ValueError('existing completion frame_count mismatch')
    if int(result.get('joint_count', -1)) != 17:
        raise ValueError('existing completion joint_count mismatch')
    completed = np.asarray(result.get('completed_keypoint'))
    expected_shape = (len(query['sample_order']), 64, 17, 3)
    if completed.shape != expected_shape or not np.isfinite(completed).all():
        raise ValueError(
            'existing completion completed_keypoint is invalid: {}'.format(
                completed.shape
            )
        )
    return result


def _summary_payload(
    plan, records, status_path, summary_path, started_at,
    provenance_warnings=None,
):
    counts = {}
    for record in records:
        status = str(record.get('status'))
        counts[status] = counts.get(status, 0) + 1
    return {
        'format': 'sars_inter_external_completion_series_summary',
        'version': 1,
        'plan_hash': plan['plan_hash'],
        'job_count': int(len(plan['jobs'])),
        'planned_count': int(counts.get('planned', 0)),
        'running_count': int(counts.get('running', 0)),
        'completed_count': int(counts.get('completed', 0)),
        'failed_count': int(counts.get('failed', 0)),
        'skipped_existing_count': int(counts.get('skipped_existing', 0)),
        'started_at_unix': float(started_at),
        'updated_at_unix': float(time.time()),
        'status_path': status_path,
        'summary_path': summary_path,
        'provenance_warnings': list(provenance_warnings or []),
        'records': copy.deepcopy(records),
    }


def _save_state(
    plan, records, status_path, summary_path, started_at,
    provenance_warnings=None,
):
    summary = _summary_payload(
        plan, records, status_path, summary_path, started_at,
        provenance_warnings=provenance_warnings,
    )
    status_payload = {
        'format': 'sars_inter_external_completion_series_status',
        'version': 1,
        'plan_hash': plan['plan_hash'],
        'updated_at_unix': summary['updated_at_unix'],
        'records': copy.deepcopy(records),
    }
    _atomic_json_dump(status_path, status_payload)
    _atomic_json_dump(summary_path, summary)
    return summary


def _progress(index, total, job_id, status, output_path):
    print('[{}/{}] job_id={} status={} output={}'.format(
        index, total, job_id, status, output_path
    ))


def _verify_plan_unchanged(plan_path, expected_hash):
    current = load_run_plan(plan_path)
    if current['plan_hash'] != expected_hash:
        raise RuntimeError('external completion run-plan changed during execution')


def _validate_plan_runtime_profile(method_profile, provenance):
    """严格校验实验资产，并将 Git revision 差异降级为提示。"""
    if str(method_profile.get('method_name') or '') != 'skeleton_in_context':
        raise ValueError('run-plan method_name is not skeleton_in_context')
    revision_pairs = (
        (
            'repository_url',
            provenance['adapter_fork']['repository_url'],
            'adapter_repository_url_differs_from_plan',
        ),
        (
            'repository_commit',
            provenance['adapter_fork']['commit'],
            'adapter_commit_differs_from_plan',
        ),
        (
            'upstream_repository_url',
            provenance['official_upstream']['repository_url'],
            'upstream_repository_url_differs_from_plan',
        ),
        (
            'upstream_repository_commit',
            provenance['official_upstream']['commit'],
            'upstream_commit_differs_from_plan',
        ),
    )
    warnings = []
    for field_name, actual_value, warning_name in revision_pairs:
        expected_value = str(method_profile.get(field_name) or '').strip()
        if expected_value and expected_value != str(actual_value):
            warnings.append(warning_name)
    adapter = provenance.get('adapter_fork') or {}
    if adapter.get('repository_dirty') is True:
        warnings.append('repository_worktree_dirty')
    if not adapter.get('commit') or adapter.get('commit') == 'unknown':
        warnings.append('repository_identity_unavailable')
    expected_checkpoint = str(
        method_profile.get('checkpoint_sha256') or ''
    ).strip().lower()
    actual_checkpoint = str(
        provenance['checkpoint'].get('sha256') or ''
    ).strip().lower()
    if expected_checkpoint != actual_checkpoint:
        raise ValueError(
            'run-plan checkpoint mismatch: expected={}, runtime={}'.format(
                expected_checkpoint, actual_checkpoint
            )
        )
    return sorted(set(warnings))


def run_completion_series(config, completion_runner=run_completion):
    resolved = copy.deepcopy(dict(config))
    plan_path = _absolute_path(resolved.get('plan_path') or '')
    output_root = _absolute_path(resolved.get('output_root') or '')
    if not os.path.isfile(plan_path):
        raise FileNotFoundError(plan_path)
    if not output_root or '<' in output_root or '>' in output_root:
        raise ValueError('output_root still contains a placeholder')
    plan = load_run_plan(plan_path)
    original_plan_hash = plan['plan_hash']
    os.makedirs(output_root, exist_ok=True)
    status_path = os.path.join(output_root, 'completion_series_status.json')
    summary_path = os.path.join(output_root, 'completion_series_summary.json')
    started_at = time.time()
    records = []
    total = len(plan['jobs'])

    if bool(resolved.get('dry_run', True)):
        checkpoint_source = _resolve_runtime_paths(resolved)
        expected_checkpoint = str(
            plan['method_profile'].get('checkpoint_sha256') or ''
        ).strip().lower()
        if expected_checkpoint != checkpoint_source['checkpoint_sha256']:
            raise ValueError(
                'run-plan checkpoint mismatch during dry-run preflight'
            )
        for index, job in enumerate(plan['jobs'], 1):
            _verify_plan_unchanged(plan_path, original_plan_hash)
            output_path = os.path.join(
                output_root, *job['result_relative_path'].split('/')
            )
            record = {
                'job_id': job['job_id'],
                'status': 'planned',
                'query_hash': job['query_hash'],
                'output_path': output_path,
                'checkpoint_source_mode': checkpoint_source[
                    'checkpoint_source_mode'
                ],
                'checkpoint_sha256': checkpoint_source[
                    'checkpoint_sha256'
                ],
                'checkpoint_identity_manifest_sha256': (
                    checkpoint_source.get(
                        'checkpoint_identity_manifest_sha256'
                    )
                ),
                'error': None,
            }
            records.append(record)
            _progress(index, total, job['job_id'], 'planned', output_path)
        _verify_plan_unchanged(plan_path, original_plan_hash)
        return _save_state(
            plan, records, status_path, summary_path, started_at
        )

    shared_runtime = {}
    prepare_completion_runtime(
        resolved, runtime=shared_runtime, load_model=False
    )
    expected_provenance = shared_runtime['provenance']
    provenance_warnings = _validate_plan_runtime_profile(
        plan['method_profile'], expected_provenance
    )
    strict = bool(resolved.get('strict', True))
    continue_on_error = bool(resolved.get('continue_on_error', False))
    resume = bool(resolved.get('resume', True))

    for index, job in enumerate(plan['jobs'], 1):
        _verify_plan_unchanged(plan_path, original_plan_hash)
        output_path = os.path.join(
            output_root, *job['result_relative_path'].split('/')
        )
        query = load_completion_query(job['query_path'])
        record = {
            'job_id': job['job_id'],
            'status': 'running',
            'query_hash': job['query_hash'],
            'output_path': output_path,
            'error': None,
        }
        if resume and os.path.isfile(output_path):
            try:
                existing = _validate_existing_result(
                    output_path, query, expected_provenance
                )
                record.update({
                    'status': 'skipped_existing',
                    'result_hash': existing['result_hash'],
                })
                records.append(record)
                _progress(
                    index, total, job['job_id'],
                    'skipped_existing', output_path,
                )
                _save_state(
                    plan, records, status_path, summary_path, started_at,
                    provenance_warnings=provenance_warnings,
                )
                continue
            except Exception as error:
                record['resume_validation_error'] = '{}: {}'.format(
                    type(error).__name__, error
                )

        records.append(record)
        _progress(index, total, job['job_id'], 'running', output_path)
        _save_state(
            plan, records, status_path, summary_path, started_at,
            provenance_warnings=provenance_warnings,
        )
        try:
            job_config = copy.deepcopy(resolved)
            job_config.update({
                'dry_run': False,
                'query_path': job['query_path'],
                'output_path': output_path,
            })
            completion_runner(job_config, runtime=shared_runtime)
            _verify_plan_unchanged(plan_path, original_plan_hash)
            result = _validate_existing_result(
                output_path, query, expected_provenance
            )
            record.update({
                'status': 'completed',
                'result_hash': result['result_hash'],
                'error': None,
            })
            _progress(index, total, job['job_id'], 'completed', output_path)
        except Exception as error:
            record.update({
                'status': 'failed',
                'error': '{}: {}'.format(type(error).__name__, error),
            })
            _progress(index, total, job['job_id'], 'failed', output_path)
            _save_state(
                plan, records, status_path, summary_path, started_at,
                provenance_warnings=provenance_warnings,
            )
            if strict and not continue_on_error:
                raise
            continue
        _save_state(
            plan, records, status_path, summary_path, started_at,
            provenance_warnings=provenance_warnings,
        )

    _verify_plan_unchanged(plan_path, original_plan_hash)
    verify_completion_runtime_assets(resolved, shared_runtime)
    return _save_state(
        plan, records, status_path, summary_path, started_at,
        provenance_warnings=provenance_warnings,
    )


def main():
    summary = run_completion_series(build_direct_run_config())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()


__all__ = ['build_direct_run_config', 'run_completion_series']
