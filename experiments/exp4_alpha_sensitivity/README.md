## 实验 4：`alpha` 敏感性

本实验考察 `adv_alpha` 对准确率和特征选择稳定性的影响。

运行示例：

```bash
python experiments/exp4_alpha_sensitivity/run_alpha_sweep.py \
  --dataset MICE \
  --runs 5 \
  --alphas 0.0,0.2,0.4,0.6,0.8,1.0 \
  --output-dir outputs/exp4_alpha_sensitivity
```
