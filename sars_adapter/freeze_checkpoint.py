"""将选定的 SiC checkpoint 冻结为可跨设备复制的正式 bundle。"""

from __future__ import annotations

import json

from sars_adapter.checkpoint_identity import freeze_checkpoint_bundle


def build_direct_run_config():
    return {
        "dry_run": True,  # True=只检查并显示哈希；False=复制文件并写入身份清单。
        "checkpoint_path": "<SIC_CHECKPOINT_PATH>",  # 可选择训练中的任意 MC-only checkpoint。
        "source_config": "<SIC_EFFECTIVE_CONFIG_PATH>",  # 必须是该 checkpoint 同一次训练保存的 effective_config.yaml。
        "output_dir": "<FROZEN_CHECKPOINT_BUNDLE_DIR>",  # 正式 bundle 输出目录，可整体复制到其它电脑。
        "checkpoint_identity_policy": "require_mc_only",  # 正式实验固定要求 MC-only single-task checkpoint。
        "overwrite": False,  # 兼容旧配置；不同正式 bundle 始终拒绝原地覆盖，同内容可复用。
    }


def main():
    summary = freeze_checkpoint_bundle(build_direct_run_config())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()


__all__ = ["build_direct_run_config", "freeze_checkpoint_bundle", "main"]
