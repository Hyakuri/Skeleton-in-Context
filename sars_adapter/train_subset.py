"""使用选定官方任务数据执行可审计的 SiC 单任务或多任务训练。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.utils.tools import get_config
from train import set_random_seed, train_with_config

OFFICIAL_TASKS = ['PE', 'MP', 'MC', 'FPE']
OFFICIAL_DATASETS = {
    'PE': 'H36M',
    'MP': 'AMASS',
    'MC': '3DPW_MC',
    'FPE': 'H36M_FPE',
}


def build_direct_run_config():
    """集中放置需要手动调整的训练参数。"""
    return {
        'dry_run': True,  # True=只检查并打印配置；False=开始真实 GPU 训练。
        'source_config': os.path.join(PROJECT_ROOT, 'configs', 'default.yaml'),  # 官方基础配置。
        'data_root': r'K:\ExternalCompletionBaselines\Skeleton-in-Context\data',  # 官方 ready-to-use 数据根目录；只要求当前启用任务的目录存在。
        'checkpoint_root': r'K:\ExternalCompletionBaselines\Skeleton-in-Context\checkpoints',  # checkpoint 与训练 manifest 输出根目录。
        'run_name': 'sic_mc_{timestamp}',  # 支持 {timestamp}；建议在名称中记录本次任务范围。
        'tasks': ['MC'],  # 可填写官方任务的有序子集；本项目补值基线推荐只使用 MC（论文中的 Joint Completion）。
        'train_sample_limits': {  # 只填写当前启用任务；None=使用该任务全部训练数据，整数=确定性抽取指定数量。
            'MC': None,
        },
        'subset_seed': 42,  # 控制各任务文件子集选择，写入 training_subset_manifest.json。
        'epochs': 120,  # 子集正式训练 epoch 数；首次运行建议先改为 1。
        'batch_size': 32,  # 单 GPU 物理 batch size；正式值应由本机 smoke test 决定。
        'test_batch_size': 256,  # 最终官方任务评价 batch size。
        'learning_rate': 0.0002,  # 沿用官方初始学习率。
        'num_workers': 0,  # Windows 首次 smoke 建议 0；正式训练可尝试 2 或 4，但必须关闭 persistent_workers。
        'pin_memory': True,  # CUDA 训练时锁页内存开关。
        'prefetch_factor': 2,  # 仅 num_workers>0 时生效。
        'persistent_workers': False,  # 严格可恢复训练必须为 False，确保每个 epoch 重新派生确定性 worker seed。
        'no_eval': True,  # 子集训练期间跳过昂贵评价；完成后再用官方 evaluate 单独评价。
        'seed': 42,  # 模型初始化、DataLoader 与训练随机性的主 seed。
    }


def _expand_timestamp(value):
    token = datetime.now().strftime('%Y%m%d%H%M%S')
    return str(value).replace('{timestamp}', token)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _require_real_path(path, field_name):
    value = os.path.abspath(os.path.expanduser(str(path)))
    if '<' in value or '>' in value:
        raise ValueError(f'{field_name} still contains a placeholder: {path}')
    return value


def resolve_training_config(config):
    resolved = copy.deepcopy(dict(config))
    resolved['tasks'] = [str(task) for task in resolved.get('tasks', [])]
    if not resolved['tasks']:
        raise ValueError('tasks must contain at least one official task')
    unsupported = [task for task in resolved['tasks'] if task not in OFFICIAL_TASKS]
    if unsupported:
        raise ValueError(f'unsupported tasks: {unsupported}; supported={OFFICIAL_TASKS}')
    if len(set(resolved['tasks'])) != len(resolved['tasks']):
        raise ValueError('tasks must not contain duplicates')
    expected_order = [task for task in OFFICIAL_TASKS if task in resolved['tasks']]
    if resolved['tasks'] != expected_order:
        raise ValueError(f'tasks must follow the official order {OFFICIAL_TASKS}')
    limits = resolved.get('train_sample_limits') or {}
    if set(limits) != set(resolved['tasks']):
        raise ValueError('train_sample_limits must define exactly the active tasks')
    for task, limit in limits.items():
        if limit is not None and int(limit) <= 0:
            raise ValueError(f'train_sample_limits[{task}] must be positive or None')
    if (
        int(resolved.get('num_workers', 0)) > 0
        and bool(resolved.get('persistent_workers', False))
    ):
        raise ValueError(
            'persistent_workers must be False for strict resumable training'
        )
    resolved['source_config'] = _require_real_path(
        resolved['source_config'], 'source_config'
    )
    resolved['data_root'] = _require_real_path(resolved['data_root'], 'data_root')
    resolved['checkpoint_root'] = _require_real_path(
        resolved['checkpoint_root'], 'checkpoint_root'
    )
    resolved['run_name'] = _expand_timestamp(resolved['run_name'])
    resolved['checkpoint_dir'] = os.path.join(
        resolved['checkpoint_root'], resolved['run_name']
    )
    if not os.path.isfile(resolved['source_config']):
        raise FileNotFoundError(resolved['source_config'])
    for task in resolved['tasks']:
        folder = OFFICIAL_DATASETS[task]
        for split in ('train', 'test'):
            path = os.path.join(resolved['data_root'], folder, split)
            if not os.path.isdir(path):
                raise FileNotFoundError(f'missing official {task}/{split}: {path}')
    resolved['task_scope'] = (
        'single_task' if len(resolved['tasks']) == 1 else 'multi_task'
    )
    return resolved


def build_effective_args(config):
    args = get_config(config['source_config'])
    args.tasks = list(config['tasks'])
    args.full_data.root_path = config['data_root']
    args.use_partial_data = False
    args.data = copy.deepcopy(args.full_data)
    args.train_sample_limits = copy.deepcopy(config['train_sample_limits'])
    args.subset_seed = int(config['subset_seed'])
    args.epochs = int(config['epochs'])
    args.batch_size = int(config['batch_size'])
    args.test_batch_size = int(config['test_batch_size'])
    args.learning_rate = float(config['learning_rate'])
    args.num_workers = int(config['num_workers'])
    args.pin_memory = bool(config['pin_memory'])
    args.prefetch_factor = int(config['prefetch_factor'])
    args.persistent_workers = bool(config['persistent_workers'])
    args.no_eval = bool(config['no_eval'])
    args.seed = int(config['seed'])
    args.strict_training_identity = True
    return args


def run_subset_training(config):
    resolved = resolve_training_config(config)
    args = build_effective_args(resolved)
    preview = {
        'status': 'planned' if resolved.get('dry_run', True) else 'running',
        'checkpoint_dir': resolved['checkpoint_dir'],
        'task_scope': resolved['task_scope'],
        'tasks': list(args.tasks),
        'train_sample_limits': _plain(args.train_sample_limits),
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'seed': int(resolved['seed']),
    }
    if resolved.get('dry_run', True):
        return preview
    os.makedirs(resolved['checkpoint_dir'], exist_ok=False)
    effective_path = os.path.join(resolved['checkpoint_dir'], 'effective_config.yaml')
    with open(effective_path, 'w') as stream:
        yaml.safe_dump(_plain(args), stream, sort_keys=False)
    with open(effective_path, 'rb') as stream:
        preview['effective_config_sha256'] = hashlib.sha256(
            stream.read()
        ).hexdigest()
    set_random_seed(int(resolved['seed']))
    opts = SimpleNamespace(
        checkpoint=resolved['checkpoint_dir'],
        config=effective_path,
        resume='',
        evaluate='',
        seed=int(resolved['seed']),
    )
    train_with_config(args, opts)
    preview['status'] = 'completed'
    return preview


def main():
    summary = run_subset_training(build_direct_run_config())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    main()
