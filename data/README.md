## 数据目录

推荐结构：

```text
data/
|- raw/
|  |- sers/
|  |- mice/
|  |- isolet/
|  |- coil-20-proc/
|  |- activity/
|  |- torchvision/
|- processed/
|- README.md
```

说明：

- `raw/` 存放原始数据，不建议直接纳入公开仓库版本控制
- `processed/` 存放缓存、中间结果和切分文件
- 如果设置了 `LASSONET_DATA_DIR`，代码会优先从该目录读取数据

## 各数据集期望文件

SERS:

- `sers/HealthyControl0.csv`
- `sers/LungCancer0.csv`

MICE:

- `mice/Data_Cortex_Nuclear.csv`

ISOLET:

- `isolet/isolet1234.data`
- `isolet/isolet5.data`

COIL:

- `coil-20-proc/obj1__0.png` 等图像文件

Activity:

- `activity/final_X_train.txt`
- `activity/final_X_test.txt`
- `activity/final_y_train.txt`
- `activity/final_y_test.txt`

Torchvision:

- `torchvision/` 由 `MNIST` 和 `FashionMNIST` 自动下载生成

## 公开分发提醒

`data/sers/` 当前存在已提交的 CSV 文件，但这与“默认不随仓库分发原始数据”的目标不一致。

正式开源前请确认：

- 这些 CSV 是否允许重新分发
- 是否涉及患者隐私或伦理审批约束
- 如果不能公开，建议移除仓库内文件并改为外部下载链接，例如 Zenodo 或 Figshare
