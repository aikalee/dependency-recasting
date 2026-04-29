import json
import torch
from torch.utils.data import Dataset


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class StructuredTokenFixDataset(Dataset):
    """
    examples format:

    [
        {
            "source": [
                {"token": "ADV", "left": [...], "right": [...]},
                ...
            ],
            "target": [
                {"token": "ADV", "left": [...], "right": [...]},
                ...
            ],
        },
        ...
    ]

    source left/right = input features
    target left/right = target labels
    """

    def __init__(self, examples, token2id, left2id, right2id):
        self.examples = examples
        self.token2id = token2id
        self.left2id = left2id
        self.right2id = right2id

        self.unk_id = token2id["<UNK>"]
        self.num_left = len(left2id)
        self.num_right = len(right2id)

    def __len__(self):
        return len(self.examples)

    def encode_token(self, token):
        return self.token2id.get(token, self.unk_id)

    def encode_ids(self, labels, label2id):
        return [label2id[x] for x in labels]

    def encode_multihot(self, labels, label2id, size):
        vec = [0] * size
        for x in labels:
            vec[label2id[x]] = 1
        return vec

    def __getitem__(self, idx):
        ex = self.examples[idx]
        source = ex["source"]
        target = ex["target"]

        assert len(source) == len(target), (
            f"Length mismatch at idx={idx}: "
            f"source={len(source)}, target={len(target)}"
        )

        input_ids = []
        left_ids = []
        right_ids = []

        left_labels = []
        right_labels = []

        raw_source = []
        raw_target = []

        for i, (s, t) in enumerate(zip(source, target)):
            assert s["token"] == t["token"], (
                f"Token mismatch at idx={idx}, pos={i}: "
                f"source={s['token']!r}, target={t['token']!r}"
            )

            token = s["token"]

            # input = source token + source structural features
            input_ids.append(self.encode_token(token))
            left_ids.append(self.encode_ids(s["left"], self.left2id))
            right_ids.append(self.encode_ids(s["right"], self.right2id))

            # labels = target structural features
            left_labels.append(
                self.encode_multihot(t["left"], self.left2id, self.num_left)
            )
            right_labels.append(
                self.encode_multihot(t["right"], self.right2id, self.num_right)
            )

            raw_source.append(s)
            raw_target.append(t)

        return {
            "input_ids": input_ids,
            "left_ids": left_ids,
            "right_ids": right_ids,
            "left_labels": left_labels,
            "right_labels": right_labels,
            "source": raw_source,
            "target": raw_target,
        }


def structured_token_fix_collate_fn(batch, pad_token_id=0):
    max_len = max(len(item["input_ids"]) for item in batch)

    num_left = len(batch[0]["left_labels"][0])
    num_right = len(batch[0]["right_labels"][0])

    input_ids = []
    attention_mask = []

    left_ids = []
    right_ids = []

    left_labels = []
    right_labels = []

    raw_source = []
    raw_target = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * seq_len + [0] * pad_len)

        # source features for embedding
        left_ids.append(item["left_ids"] + [[] for _ in range(pad_len)])
        right_ids.append(item["right_ids"] + [[] for _ in range(pad_len)])

        # target multi-hot labels
        left_pad = [[0] * num_left for _ in range(pad_len)]
        right_pad = [[0] * num_right for _ in range(pad_len)]

        left_labels.append(item["left_labels"] + left_pad)
        right_labels.append(item["right_labels"] + right_pad)

        raw_source.append(item["source"])
        raw_target.append(item["target"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),

        # keep nested lists because each token has variable number of features
        "left_ids": left_ids,
        "right_ids": right_ids,

        "left_labels": torch.tensor(left_labels, dtype=torch.float),
        "right_labels": torch.tensor(right_labels, dtype=torch.float),

        "source": raw_source,
        "target": raw_target,
    }