# Adversarial LassoNet

面向论文配套代码公开的 Adversarial / SAM-style LassoNet 实验仓库，包含主基准实验、SERS 二分类实验、Colored MNIST 对比和消融分析。

本仓库包含一份 vendored `lassonet` 实现，遵循 MIT License，版权归原作者所有：
Ismael Lemhadri et al.，上游项目见 <https://github.com/lasso-net/lassonet>。相关许可证已保存在 `src/models/lassonet/` 和 `src/models/lassonet_sers/`。

`src/utils/data_utils.py` 中部分表格数据加载逻辑参考了 Concrete Autoencoders：
<https://github.com/mfbalin/Concrete-Autoencoders>。

## 项目结构

```text
Adversarial-LassoNet/
|- configs/         # 实验默认配置
|- data/            # 数据说明与本地数据目录
|- docs/            # 复现与项目文档
|- experiments/     # 对外推荐的实验入口
|- outputs/         # 结果输出目录
|- scripts/         # 兼容旧工作流的脚本入口
|- src/             # 核心模型、数据与分析代码
|- pyproject.toml   # 可安装包配置
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows:

```bash
.venv\Scripts\activate
pip install -e .
```

推荐环境：

- Python 3.10 或 3.11
- PyTorch 2.x
- CUDA 11.8+（可选；CPU 也可运行，但 benchmark 会更慢）

## 数据准备

默认数据根目录是 `data/raw/`，也可以通过环境变量覆盖：

```bash
export LASSONET_DATA_DIR=/path/to/data/raw
```

Windows PowerShell:

```powershell
$env:LASSONET_DATA_DIR = "D:\\datasets\\adversarial-lassonet"
```

推荐目录结构：

```text
data/
|- raw/
|  |- sers/
|  |  |- HealthyControl0.csv
|  |  |- LungCancer0.csv
|  |- mice/
|  |  |- Data_Cortex_Nuclear.csv
|  |- isolet/
|  |  |- isolet1234.data
|  |  |- isolet5.data
|  |- coil-20-proc/
|  |- activity/
|  |  |- final_X_train.txt
|  |  |- final_X_test.txt
|  |  |- final_y_train.txt
|  |  |- final_y_test.txt
|  |- torchvision/
```

数据集获取链接与说明请见 [data/README.md](data/README.md)。

说明：

- `MNIST` 和 `FashionMNIST` 会由 `torchvision` 自动下载到 `data/raw/torchvision/`
- 当前仓库中仍保留了 `data/sers/` 示例 CSV；在正式公开前，请先确认你是否拥有重新分发权限

## 快速开始

5 分钟 smoke test：

```bash
python experiments/exp1_main_benchmark/run_sers.py --mode clean --seed 42
```

主实验：

```bash
python experiments/exp1_main_benchmark/run_benchmark.py
```

其它实验：

```bash
python experiments/exp2_spurious_colored_mnist/run_comparison.py
python experiments/exp3_ablation/run_ablation.py --datasets MICE --ablation all --runs 5
python experiments/exp4_alpha_sensitivity/run_alpha_sweep.py --dataset MICE --runs 5
```

## 输出

默认结果会写入：

- `outputs/exp1_main_benchmark/`
- `outputs/exp2_spurious_colored_mnist/`
- `outputs/exp3_ablation/`
- `outputs/exp4_alpha_sensitivity/`

## 可复现性说明

- 当前 `configs/exp1_main_benchmark/` 已接入 `run_sers.py` 和 `run_benchmark.py`
- `scripts/` 目录仍保留，用于兼容旧命令；对外使用优先选择 `experiments/`
- 若需要完全复现实验，请同时记录 Python、PyTorch、CUDA、随机种子和数据目录

## 引用

如果你使用了本仓库，请至少引用原始 LassoNet 工作，并在你的论文中注明本仓库的实验实现来源。

后续建议补充：

- `CITATION.cff`
- 论文 BibTeX
- 英文版 README

## License

本仓库主代码采用 MIT License，见 [LICENSE](LICENSE)。
