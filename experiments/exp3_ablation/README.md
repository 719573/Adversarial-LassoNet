## 实验 3：消融实验

本实验对应论文中的模块消融，分析不同正则项和对抗扰动组件对性能与特征稳定性的影响。

运行示例：

```bash
python experiments/exp3_ablation/run_ablation.py \
  --datasets MICE \
  --runs 5 \
  --ablation all \
  --save-json outputs/exp3_ablation/ablation_summary.json
```
