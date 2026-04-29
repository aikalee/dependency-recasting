import os
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.pathgen import get_edit_actions_txt_path
from src.models.structured_token_edit.load_data import read_jsonl, StructuredTokenDataset, structured_token_collate_fn
from src.models.structured_token_edit.model import StructuredTokenTransformer
from src.models.structured_token_edit.trainer import StructuredTokenTrainer


DATA_DIR = ROOT / "data" / "downstream"
ARTIFACT_DIR = ROOT / "artifacts"


def make_dataloader(examples, token2id, left2id, right2id, batch_size=8, shuffle=False):
    dataset = StructuredTokenDataset(
        examples=examples,
        token2id=token2id,
        left2id=left2id,
        right2id=right2id,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: structured_token_collate_fn(
            batch,
            pad_token_id=token2id["<PAD>"],
        ),
    )


def main():
    data = DATA_DIR / "lang=combined,pos=upos"
    train_path = data / "train.json"
    dev_path = data / "dev.json"
    vocab_path = ARTIFACT_DIR / "vocab_and_weights.json"

    artifacts = read_jsonl(vocab_path)

    token2id = artifacts["token2id"]
    left2id = artifacts["left2id"]
    right2id = artifacts["right2id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    left_pos_weight = torch.tensor(
        artifacts["left_pos_weight"],
        dtype=torch.float,
        device=device,
    )
    right_pos_weight = torch.tensor(
        artifacts["right_pos_weight"],
        dtype=torch.float,
        device=device,
    )

    train_examples = read_jsonl(train_path)
    dev_examples = read_jsonl(dev_path)

    batch_size = 8

    train_loader = make_dataloader(
        train_examples,
        token2id,
        left2id,
        right2id,
        batch_size=batch_size,
        shuffle=True,
    )

    dev_loader = make_dataloader(
        dev_examples,
        token2id,
        left2id,
        right2id,
        batch_size=batch_size,
        shuffle=False,
    )

    model_config = {
        "vocab_size": len(token2id),
        "num_left_labels": len(left2id),
        "num_right_labels": len(right2id),
        "d_model": 256,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "pad_token_id": token2id["<PAD>"],
        "max_len": 2048,
        "use_two_layer_head": True,
    }

    model = StructuredTokenTransformer(**model_config)

    trainer = StructuredTokenTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        left_label2id=left2id,
        right_label2id=right2id,
        left_pos_weight=left_pos_weight,
        right_pos_weight=right_pos_weight,
        device=device,
        lr=3e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        left_loss_weight=1.0,
        right_loss_weight=1.0,
    )

    trainer.fit(
        num_epochs=10,
        save_dir=str(ARTIFACT_DIR),
        save_best_only=True,
    )


if __name__ == "__main__":
    main()