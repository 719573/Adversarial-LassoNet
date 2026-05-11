## 实验 2：Colored MNIST 虚假关联

本实验比较 vanilla `LassoNet` 和 `adv-LassoNet` 在 Colored MNIST 上面对虚假颜色关联时的表现。

运行示例：

```bash
python experiments/exp2_spurious_colored_mnist/run_comparison.py \
  --runs 5 \
  --k 50 \
  --save-json outputs/exp2_spurious_colored_mnist/colored_mnist_report.json
```
