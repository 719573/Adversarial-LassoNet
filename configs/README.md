## 配置目录

本目录存放实验默认参数。当前已经接入的入口：

- `exp1_main_benchmark/sers.yaml`
- `exp1_main_benchmark/table2.yaml`

用法示例：

```bash
python experiments/exp1_main_benchmark/run_sers.py
python experiments/exp1_main_benchmark/run_benchmark.py
```

以上入口会先读取 YAML 默认值，再允许命令行参数覆盖。

尚未完全接入但保留目录占位的配置：

- `exp2_spurious_colored_mnist/default.yaml`
- `exp3_ablation/default.yaml`
- `exp4_alpha_sensitivity/default.yaml`

这些文件目前仍应视为 WIP。
