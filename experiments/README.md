## 论文实验入口

本目录按论文实验组织，而不是按模型脚本组织。

- `exp1_main_benchmark/`：主实验，在 SERS 和 benchmark 数据集上比较 vanilla LassoNet、adv-LassoNet、FISTA-Net、Deep-Lasso。
- `exp2_spurious_colored_mnist/`：Colored MNIST 上的虚假关联实验。
- `exp3_ablation/`：消融实验。
- `exp4_alpha_sensitivity/`：`alpha` 敏感性实验。

每个子目录下都放：

- `run_*.py`：实验主入口。
- `README.md`：该实验的运行说明。
- 可选的 `summarize.py` / `plot_*.py`：结果汇总和绘图脚本。
