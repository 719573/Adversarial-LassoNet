## Data Directory

Recommended layout:

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

Notes:

- `raw/` stores raw datasets and is generally not intended for public version control
- `processed/` stores caches, intermediate artifacts, and split files
- If `LASSONET_DATA_DIR` is set, the code reads data from that location with priority

## Expected Files Per Dataset

SERS:

- `sers/HealthyControl0.csv`
- `sers/LungCancer0.csv`

MICE:

- `mice/Data_Cortex_Nuclear.csv`

ISOLET:

- `isolet/isolet1234.data`
- `isolet/isolet5.data`

COIL:

- image files such as `coil-20-proc/obj1__0.png`

Activity:

- `activity/final_X_train.txt`
- `activity/final_X_test.txt`
- `activity/final_y_train.txt`
- `activity/final_y_test.txt`

Torchvision:

- `torchvision/` is created automatically by downloading `MNIST` and `FashionMNIST`

## Data Release Notes

`data/sers/` currently contains committed CSV files, which is inconsistent with the goal of not distributing raw data with the repository by default.

Before a public release, the following points should be reviewed:

- Whether these CSV files may be redistributed
- Whether they are subject to patient-privacy or ethics-approval constraints
- If public redistribution is not allowed, replace repository copies with external download links such as Zenodo or Figshare
