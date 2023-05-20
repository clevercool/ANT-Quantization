# BERT / BART
## Usage

You can run the following scripts to reproduce the results in the paper. The results may slightly differ from those in the paper due to version updates of some libraries.

```bash
./scripts/run_all.sh
```

## Results

| Model | CoLA  | SST-2 | MNLI | QQP | MPRC |
| :----:| :----: | :----: | :----: | :----: | :----: |
| BERT-base | 59.99 | 92.55 | 83.83 | 90.34 | 87.75 |
| BART-base | 52.69 | 94.15 | 85.54 | 91.37 | 85.29 |
| BERT-large | 63.23 | 92.43 | 84.80 | 90.11 | 87.01 |


| Model | SQuAD v1.1  | SQuAD v2.0 |
| :----:| :----: | :----: |
| BERT-base | 86.38 / 78.24 | 75.67 / 71.54 |
| BART-base | 88.07 / 79.81  | 77.70 / 74.08 |