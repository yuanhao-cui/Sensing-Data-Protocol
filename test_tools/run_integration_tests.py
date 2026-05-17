#!/usr/bin/env python3
"""
自动化集成测试脚本

对 widar、gait、xrf55、elderAL 四个数据集分别执行以下 4 种调用：
1. pipeline 直接调用（无可选参数）
2. pipeline 直接调用（可选参数：指定模型 + algorithm preset）
3. CLI 调用（无可选参数）
4. CLI 调用（可选参数：指定模型 + algorithm preset）

为控制执行时间，所有调用均固定 num_epochs=1、num_seeds=1。

用法：
    python test_tools/run_integration_tests.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 强制使用非交互式 matplotlib backend，避免在子进程/后台线程中弹出 GUI
import matplotlib
matplotlib.use("Agg")

# 将 src/ 加入当前进程路径，确保本脚本可直接导入 wsdp
SRC_PATH = str(Path(__file__).parent.parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from wsdp import pipeline
from wsdp.cli import main_cli
from wsdp.algorithms import list_presets
from wsdp.models import list_models

# =============================================================================
# 配置区
# =============================================================================
DATASETS = ["widar", "gait", "xrf55", "elderAL"]
DATA_ROOT = "./data"
OUTPUT_ROOT = "./test_script_output"
MODEL_NAME = "cnn1dmodel"
ALGORITHM_PRESET = "robust"
NUM_EPOCHS = 3
TIMEOUT_SECONDS = 600  # 单条测试超时（秒）


# =============================================================================
# 前置校验
# =============================================================================
def _validate_env():
    """校验模型名和 preset 名真实存在，防止传错参数。"""
    available_models = list_models()
    if MODEL_NAME.lower() not in available_models:
        raise ValueError(
            f"未知模型 '{MODEL_NAME}'。可用模型: {list(available_models.keys())}"
        )

    available_presets = list_presets()
    if ALGORITHM_PRESET not in available_presets:
        raise ValueError(
            f"未知 preset '{ALGORITHM_PRESET}'。可用 presets: {list(available_presets.keys())}"
        )

    print(f"[INFO] 测试使用模型: {MODEL_NAME}")
    print(f"[INFO] 测试使用 preset: {ALGORITHM_PRESET}")
    print(f"[INFO] 数据集根目录: {DATA_ROOT}")
    print(f"[INFO] 输出根目录: {OUTPUT_ROOT}")


def _dataset_exists(dataset: str) -> bool:
    """检查数据集目录是否存在且非空。"""
    p = Path(DATA_ROOT) / dataset
    return p.exists() and any(p.iterdir())


# =============================================================================
# 测试执行函数
# =============================================================================
def _run_pipeline_basic(dataset: str, output_dir: str) -> dict:
    """方式1：pipeline() 无可选参数。"""
    start = time.time()
    try:
        pipeline(
            input_path=str(Path(DATA_ROOT) / dataset),
            output_folder=output_dir,
            dataset=dataset,
            num_epochs=NUM_EPOCHS,
            num_seeds=2,
            use_cache=True,
        )
        return {"status": "PASSED", "elapsed": round(time.time() - start, 2)}
    except Exception as e:
        return {"status": "FAILED", "error": str(e), "elapsed": round(time.time() - start, 2)}


def _run_pipeline_with_options(dataset: str, output_dir: str) -> dict:
    """方式2：pipeline() 带可选参数（模型 + preset）。"""
    start = time.time()
    try:
        pipeline(
            input_path=str(Path(DATA_ROOT) / dataset),
            output_folder=output_dir,
            dataset=dataset,
            model_name=MODEL_NAME,
            algorithm_preset=ALGORITHM_PRESET,
            num_epochs=NUM_EPOCHS,
            num_seeds=2,
            use_cache=True,
        )
        return {"status": "PASSED", "elapsed": round(time.time() - start, 2)}
    except Exception as e:
        return {"status": "FAILED", "error": str(e), "elapsed": round(time.time() - start, 2)}


def _build_cli_subprocess_code(argv: list, num_epochs: int) -> str:
    """
    构造在子进程中通过 python -c 执行的代码。

    由于 CLI 当前未暴露 --seeds 参数，我们在子进程中用 mock.patch 拦截
    wsdp.cli.pipeline，在保持 CLI 参数解析完全真实的前提下，注入 num_seeds=1。
    """
    argv_repr = repr(argv)
    return f"""
import sys
sys.path.insert(0, {repr(SRC_PATH)})
sys.argv = {argv_repr}

from unittest.mock import patch
from wsdp.cli import main_cli
from wsdp.core import pipeline as real_pipeline

def _patched_pipeline(*args, **kwargs):
    kwargs["num_seeds"] = 2
    return real_pipeline(*args, **kwargs)

with patch("wsdp.cli.pipeline", side_effect=_patched_pipeline):
    main_cli()
"""


def _run_cli_basic(dataset: str, output_dir: str) -> dict:
    """方式3：CLI 无可选参数。"""
    start = time.time()
    argv = [
        "wsdp", "run",
        str(Path(DATA_ROOT) / dataset),
        output_dir,
        dataset,
        "--epochs", str(NUM_EPOCHS),
    ]
    code = _build_cli_subprocess_code(argv, NUM_EPOCHS)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            return {"status": "PASSED", "elapsed": round(time.time() - start, 2)}
        else:
            err = proc.stderr.strip()[-800:] if proc.stderr else "(no stderr)"
            return {"status": "FAILED", "error": err, "elapsed": round(time.time() - start, 2)}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "elapsed": round(time.time() - start, 2)}
    except Exception as e:
        return {"status": "FAILED", "error": str(e), "elapsed": round(time.time() - start, 2)}


def _run_cli_with_options(dataset: str, output_dir: str) -> dict:
    """方式4：CLI 带可选参数（模型 + preset）。"""
    start = time.time()
    argv = [
        "wsdp", "run",
        str(Path(DATA_ROOT) / dataset),
        output_dir,
        dataset,
        "--model", MODEL_NAME,
        "--algorithm-preset", ALGORITHM_PRESET,
        "--epochs", str(NUM_EPOCHS),
    ]
    code = _build_cli_subprocess_code(argv, NUM_EPOCHS)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            return {"status": "PASSED", "elapsed": round(time.time() - start, 2)}
        else:
            err = proc.stderr.strip()[-800:] if proc.stderr else "(no stderr)"
            return {"status": "FAILED", "error": err, "elapsed": round(time.time() - start, 2)}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "elapsed": round(time.time() - start, 2)}
    except Exception as e:
        return {"status": "FAILED", "error": str(e), "elapsed": round(time.time() - start, 2)}


# =============================================================================
# 主控逻辑
# =============================================================================
def main():
    _validate_env()
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    results = []
    test_plan = [
        ("pipeline_basic", _run_pipeline_basic),
        ("pipeline_options", _run_pipeline_with_options),
        ("cli_basic", _run_cli_basic),
        ("cli_options", _run_cli_with_options),
    ]

    for dataset in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"数据集: {dataset}")
        print(f"{'=' * 60}")

        if not _dataset_exists(dataset):
            print(f"  ⚠️  数据目录不存在或为空，跳过 4 项测试")
            for name, _ in test_plan:
                results.append({
                    "dataset": dataset,
                    "test": name,
                    "status": "SKIPPED",
                    "reason": f"{DATA_ROOT}/{dataset} not found or empty",
                })
            continue

        for idx, (name, func) in enumerate(test_plan, start=1):
            out_dir = str(Path(OUTPUT_ROOT) / dataset / name)
            print(f"  [{idx}/4] {name} ...", end=" ", flush=True)
            res = func(dataset, out_dir)
            res["dataset"] = dataset
            res["test"] = name
            results.append(res)
            print(f"{res['status']} ({res.get('elapsed', 0):.1f}s)")

    # -------------------------------------------------------------------------
    # 汇总报告
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("测试汇总")
    print(f"{'=' * 60}")

    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    timeout = sum(1 for r in results if r["status"] == "TIMEOUT")

    print(f"  通过  : {passed}")
    print(f"  失败  : {failed}")
    print(f"  跳过  : {skipped}")
    print(f"  超时  : {timeout}")
    print(f"  总计  : {len(results)}")

    report_path = Path(OUTPUT_ROOT) / "integration_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  详细报告已保存: {report_path}")

    # 失败时以非 0 退出码退出，便于 CI 集成
    if failed > 0 or timeout > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
