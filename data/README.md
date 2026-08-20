# Included CSV Data

## Training

- `train/porn_5000.csv`: z03 / classifier-head feedback 的 porn 训练集。
- `train/gore_5000.csv`: z03 / classifier-head feedback 的 gore 训练集。
- `train/ip_5_z03_filtered_23760.csv`: five-IP merged unsafe training prompts.
- `train/benign_all.csv`: benign training prompts.
- `train/single_ip/`: per-IP unsafe training subsets.

## Testing

- `test/porn_level_5.csv`: porn 评测集主版本。
- `test/gore_level_5.csv`: gore 评测集主版本。
- `test/porn_level_4.csv` / `test/porn_level_4_5.csv`: 额外 porn 评测切分。
- `test/gore_level_4.csv` / `test/gore_level_4_5.csv`: 额外 gore 评测切分。
- `test/ip_5.csv`: five-IP merged evaluation prompts.
- `test/ip_by_category/`: evaluation prompts split by IP.
- `test/benign_200_train_disjoint.csv`: benign evaluation prompts separated from the training benign set.
- `test/benign_200.csv`: original benign evaluation subset.

## Optional Preservation Data

`optional_related_benign/` contains related benign prompts for IP preservation experiments.
Pass one or more files with `--extra_benign_csvs` when training.
