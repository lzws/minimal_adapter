# Included CSV Data

## Training

- `train/ip_5_z03_filtered_23760.csv`: five-IP merged unsafe training prompts.
- `train/benign_all.csv`: benign training prompts.
- `train/single_ip/`: per-IP unsafe training subsets.

## Testing

- `test/ip_5.csv`: five-IP merged evaluation prompts.
- `test/ip_by_category/`: evaluation prompts split by IP.
- `test/benign_200_train_disjoint.csv`: benign evaluation prompts separated from the training benign set.
- `test/benign_200.csv`: original benign evaluation subset.

## Optional Preservation Data

`optional_related_benign/` contains related benign prompts for IP preservation experiments.
Pass one or more files with `--extra_benign_csvs` when training.
