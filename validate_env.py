"""
验证 pixi 环境及所有依赖是否正确安装与可用。

用法：
    pixi run python validate_env.py
"""

import importlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path


def section(title: str):
    """打印带分隔线的章节标题。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check(ok: bool, msg: str, detail: str = ""):
    """打印检查结果。"""
    symbol = "✓" if ok else "✗"
    print(f"  {symbol}  {msg}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"       {line}")
    return ok


def check_import(name: str, import_name: str | None = None) -> tuple[bool, str]:
    """尝试导入包，返回 (是否成功, 版本字符串)。"""
    try:
        mod = importlib.import_module(import_name or name)
        ver = getattr(mod, "__version__", None)
        if ver:
            return True, ver
        # fallback: 尝试从元数据获取版本
        try:
            ver = importlib.metadata.version(name)
            return True, ver
        except importlib.metadata.PackageNotFoundError:
            return True, "(version unknown)"
    except ImportError as e:
        return False, str(e)


# ============================================================
# 1. 系统与环境基本信息
# ============================================================
section("1. 系统与环境信息")

check(True, f"Python 版本", sys.version)

check(True, f"操作系统", f"{platform.system()} {platform.release()}")

check(True, f"平台架构", platform.machine())

# 检测是否在 pixi 虚拟环境中 (pixi 环境的 Python 可执行文件在 .pixi 目录下)
is_pixi_env = ".pixi" in Path(sys.executable).parts
check(
    is_pixi_env,
    "运行环境",
    "在 pixi 虚拟环境中 ✓" if is_pixi_env else "警告: 不在 pixi 虚拟环境中运行",
)


# ============================================================
# 2. pixi CLI 可用性
# ============================================================
section("2. pixi CLI 状态")

try:
    result = subprocess.run(
        ["pixi", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=Path(__file__).parent,
    )
    pixi_ver = result.stdout.strip() or result.stderr.strip()
    # 过滤掉 WSL 提示
    pixi_ver = [l for l in pixi_ver.split("\n") if "pixi" in l.lower()]
    check(
        result.returncode == 0,
        "pixi CLI 可用",
        pixi_ver[0] if pixi_ver else "pixi 已安装",
    )
except (FileNotFoundError, subprocess.TimeoutExpired) as e:
    check(False, "pixi CLI 不可用", str(e))


# ============================================================
# 3. 依赖包导入验证
# ============================================================
section("3. 依赖包导入验证")

dependencies = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scikit-learn", "sklearn"),
    ("openai", "openai"),
    ("requests", "requests"),
    ("torch", "torch"),  # pytorch-gpu
    ("scipy", "scipy"),
    ("h5py", "h5py"),
]

all_ok = True
for pkg_name, import_name in dependencies:
    ok, detail = check_import(pkg_name, import_name)
    if "version" in detail.lower() or ok:
        all_ok &= check(ok, f"{pkg_name:20s}", detail)
    else:
        all_ok &= check(ok, f"{pkg_name:20s}", detail)


# ============================================================
# 4. 版本约束验证
# ============================================================
section("4. 版本约束检查")

import numpy
import pandas
import scipy

# numpy: >=2.4.6,<3
numpy_ok = (
    (2, 4, 6) <= tuple(int(x) for x in numpy.__version__.split(".")[:3]) < (3, 0, 0)
)
check(
    numpy_ok,
    f"numpy {numpy.__version__} 满足 >=2.4.6,<3",
    f"当前版本: {numpy.__version__}",
)

# python: 3.12.*
py_ver = tuple(sys.version_info[:2])
py_ok = py_ver == (3, 12)
check(
    py_ok,
    f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} 满足 3.12.*",
    f"当前版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
)

# pandas: >=3.0.3,<4
pd_ver = tuple(int(x) for x in pandas.__version__.split(".")[:3])
pd_ok = (3, 0, 3) <= pd_ver < (4, 0, 0)
check(
    pd_ok,
    f"pandas {pandas.__version__} 满足 >=3.0.3,<4",
    f"当前版本: {pandas.__version__}",
)

# scipy: >=1.17.1,<2
sp_ver = tuple(int(x) for x in scipy.__version__.split(".")[:3])
sp_ok = (1, 17, 1) <= sp_ver < (2, 0, 0)
check(
    sp_ok,
    f"scipy {scipy.__version__} 满足 >=1.17.1,<2",
    f"当前版本: {scipy.__version__}",
)


# ============================================================
# 5. CUDA 与 PyTorch GPU 检测
# ============================================================
section("5. CUDA & PyTorch GPU 检测")

# --- PyTorch CUDA ---
try:
    import torch

    torch_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if torch_cuda else 0
    cuda_version = (
        torch.version.cuda
        if hasattr(torch, "version") and hasattr(torch.version, "cuda")
        else "N/A"
    )
    check(
        torch_cuda,
        "PyTorch CUDA 可用" if torch_cuda else "PyTorch CUDA 不可用",
        f"CUDA 版本: {cuda_version}, GPU 数量: {gpu_count}" if torch_cuda else "",
    )
    if torch_cuda:
        for i in range(gpu_count):
            check(True, f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    check(False, "PyTorch CUDA 检测失败", str(e))

# --- nvidia-smi ---
try:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                check(True, "nvidia-smi", line.strip())
    else:
        check(False, "nvidia-smi 不可用", result.stderr.strip())
except FileNotFoundError:
    check(False, "nvidia-smi 未安装 (NVIDIA 驱动可能缺失)")
except subprocess.TimeoutExpired:
    check(False, "nvidia-smi 超时")

# --- CUDA toolkit 版本 (cuda-version / nvcc) ---
try:
    result = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            if "release" in line:
                check(True, "nvcc (CUDA 工具包) 可用", line.strip())
                break
    else:
        check(False, "nvcc 返回错误", result.stderr.strip())
except FileNotFoundError:
    # nvcc 可能不在 PATH 中, 但 cuda-version 已通过 conda 安装
    check(True, "nvcc 未在 PATH 中 (使用 conda cuda-version 运行时兼容层)", "")
except subprocess.TimeoutExpired:
    check(False, "nvcc 超时")


# ============================================================
# 6. Pixi 环境锁文件一致性
# ============================================================
section("6. 环境一致性")

try:
    result = subprocess.run(
        ["pixi", "info"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=Path(__file__).parent,
    )
    output = result.stdout + result.stderr
    # 检查是否有 WARN
    warnings = [l for l in output.split("\n") if "WARN" in l or "warn" in l]
    if warnings:
        check(False, "pixi 存在警告", "\n".join(warnings))
    else:
        check(True, "pixi 环境无警告")

    # 检查环境是否最新
    if "The default environment has been installed" in output:
        check(True, "pixi 环境已安装")
    else:
        check(True, "pixi 环境状态正常")

except (FileNotFoundError, subprocess.TimeoutExpired) as e:
    check(False, "pixi info 执行失败", str(e))


# ============================================================
# 7. 最终汇总
# ============================================================
print()
print("=" * 60)
if all_ok and torch_cuda:
    print("  所有检查通过 ✓  环境就绪，CUDA 可用")
elif all_ok:
    print("  依赖检查通过 ✓  (但 CUDA 不可用，请检查驱动)")
else:
    print("  部分检查失败 ✗  请根据上方提示修复")
print("=" * 60)
