# Stanza Dependency Parsing Experiments

|   | Settings                               | UAS   | LAS   |
|---|----------------------------------------|-------|-------|
| 1 | Frozen BERT + charLM + Pretrain        | 93.72 | 92.21 |
| 2 | Finetuning BERT + charLM + Pretrain    | 91.61 | 89.98 |
| 3 | Frozen BERT + charLM                   | 93.39 | 91.87 |
| 4 | Frozen BERT + Pretrain                 | 93.53 | 91.96 |
| 5 | Frozen BERT only                       | 93.83 | 92.29 |
| 6 | Finetuning BERT only                   | 89.92 | 88.07 |
| 7 | charLM only                            | 91.82 | 90.29 |

# Stanza Constituency Parsing Experiments (20 epochs)
|   | Settings                               | UAS   | LAS   |
|---|----------------------------------------|-------|-------|
| 1 | Frozen BERT + charLM                   | 94.67 | 93.40 |
| 2 | Finetuning BERT + charLM               | 95.83 | 94.70 |
| 3 | Frozen BERT only                       | 94.61 | 93.30 |
| 4 | Finetuning BERT only                   | 95.79 | 94.75 |
| 5 | charLM only                            | 91.38 | 89.96 |


# Remarks
1. Since fine-tuning BERT consistently underperformed frozen BERT, both when used alone and in combination with other components, we did not run experiments with fine-tuned BERT combined with charLM or pretrained embeddings.
2. The constituency parser does not support the --no_pretrain option.