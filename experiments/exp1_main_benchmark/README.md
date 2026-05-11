## 实验 1：主实验

本实验对应论文中的主结果：

- SERS 数据上的 `LassoNet` 与 `adv-LassoNet` 对比。
- benchmark 数据集上 `LassoNet`、`adv-LassoNet`、`FISTA-Net`、`Deep-Lasso` 对比。

运行示例：

```bash
python experiments/exp1_main_benchmark/run_sers.py
python experiments/exp1_main_benchmark/run_benchmark.py --datasets table2 --runs 5 --k 50
```

建议将结果输出到：

- `outputs/exp1_main_benchmark/`
