import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


class StructuredTokenTrainer:
    def __init__(
        self,
        model,
        train_loader,
        left_label2id,
        right_label2id,
        left_pos_weight: Optional[torch.Tensor] = None,
        right_pos_weight: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dev_loader=None,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        grad_clip: float = 1.0,
        left_loss_weight: float = 1.0,
        right_loss_weight: float = 1.0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.left_label2id = left_label2id
        self.right_label2id = right_label2id
        self.grad_clip = grad_clip
        self.left_loss_weight = left_loss_weight
        self.right_loss_weight = right_loss_weight

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if left_pos_weight is not None:
            left_pos_weight = left_pos_weight.to(self.device)
        if right_pos_weight is not None:
            right_pos_weight = right_pos_weight.to(self.device)

        self.left_loss_fn = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=left_pos_weight,
        )
        self.right_loss_fn = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=right_pos_weight,
        )

        self.config = {
            "device": str(self.device),
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "left_loss_weight": left_loss_weight,
            "right_loss_weight": right_loss_weight,
        }

        self.history = {
            "train_loss": [],
            "train_left_loss": [],
            "train_right_loss": [],
            "dev_loss": [],
            "dev_left_loss": [],
            "dev_right_loss": [],
        }

        model_config = getattr(self.model, "config", None)
        if model_config is not None:
            print(
                "Model config",
                json.dumps(model_config, indent=2),
                "\nTraining config",
                json.dumps(self.config, indent=2),
            )
        else:
            print("Training config", json.dumps(self.config, indent=2))

    def compute_losses(self, outputs, batch):
        left_logits = outputs["left_logits"]    # [B, T, L]
        right_logits = outputs["right_logits"]  # [B, T, R]

        left_labels = batch["left_labels"].to(self.device).float()      # [B, T, L]
        right_labels = batch["right_labels"].to(self.device).float()    # [B, T, R]
        attention_mask = batch["attention_mask"].to(self.device).bool() # [B, T]

        left_loss_raw = self.left_loss_fn(left_logits, left_labels)      # [B, T, L]
        right_loss_raw = self.right_loss_fn(right_logits, right_labels)  # [B, T, R]

        if attention_mask.any():
            left_loss = left_loss_raw[attention_mask].mean()
            right_loss = right_loss_raw[attention_mask].mean()
        else:
            left_loss = left_logits.new_tensor(0.0)
            right_loss = right_logits.new_tensor(0.0)

        total_loss = (
            self.left_loss_weight * left_loss
            + self.right_loss_weight * right_loss
        )

        return {
            "loss": total_loss,
            "left_loss": left_loss.detach(),
            "right_loss": right_loss.detach(),
        }

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_left_loss = 0.0
        total_right_loss = 0.0

        for batch in tqdm(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                left_ids=batch["left_ids"],
                right_ids=batch["right_ids"],
                attention_mask=attention_mask,
            )
            losses = self.compute_losses(outputs, batch)
            loss = losses["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            total_left_loss += losses["left_loss"].item()
            total_right_loss += losses["right_loss"].item()

        n = len(self.train_loader)
        avg_loss = total_loss / n
        avg_left_loss = total_left_loss / n
        avg_right_loss = total_right_loss / n

        self.history["train_loss"].append(avg_loss)
        self.history["train_left_loss"].append(avg_left_loss)
        self.history["train_right_loss"].append(avg_right_loss)

        return {
            "loss": avg_loss,
            "left_loss": avg_left_loss,
            "right_loss": avg_right_loss,
        }

    @torch.no_grad()
    def evaluate(self, dataloader=None):
        dataloader = dataloader or self.dev_loader
        if dataloader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_left_loss = 0.0
        total_right_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                left_ids=batch["left_ids"],
                right_ids=batch["right_ids"],
                attention_mask=attention_mask,
            )
            losses = self.compute_losses(outputs, batch)

            total_loss += losses["loss"].item()
            total_left_loss += losses["left_loss"].item()
            total_right_loss += losses["right_loss"].item()

        n = len(dataloader)
        avg_loss = total_loss / n
        avg_left_loss = total_left_loss / n
        avg_right_loss = total_right_loss / n

        self.history["dev_loss"].append(avg_loss)
        self.history["dev_left_loss"].append(avg_left_loss)
        self.history["dev_right_loss"].append(avg_right_loss)

        return {
            "loss": avg_loss,
            "left_loss": avg_left_loss,
            "right_loss": avg_right_loss,
        }

    def fit(self, num_epochs: int, save_dir: Optional[str] = None, save_best_only: bool = True):
        best_dev_loss = float("inf")

        if save_dir is not None:
            self.save_config(save_dir)

        for epoch in range(1, num_epochs + 1):
            print(f"Starting Epoch {epoch}")

            train_stats = self.train_one_epoch()
            dev_stats = self.evaluate()

            if dev_stats is None:
                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_stats['loss']:.4f} "
                    f"(left={train_stats['left_loss']:.4f}, right={train_stats['right_loss']:.4f})"
                )
                if save_dir is not None and not save_best_only:
                    self.save_checkpoint(save_dir, f"epoch_{epoch}.pt", epoch, train_stats, None)
            else:
                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_stats['loss']:.4f} "
                    f"(left={train_stats['left_loss']:.4f}, right={train_stats['right_loss']:.4f}) | "
                    f"dev_loss={dev_stats['loss']:.4f} "
                    f"(left={dev_stats['left_loss']:.4f}, right={dev_stats['right_loss']:.4f})"
                )

                if save_dir is not None:
                    if save_best_only:
                        if dev_stats["loss"] < best_dev_loss:
                            best_dev_loss = dev_stats["loss"]
                            self.save_checkpoint(save_dir, "best.pt", epoch, train_stats, dev_stats)
                    else:
                        self.save_checkpoint(save_dir, f"epoch_{epoch}.pt", epoch, train_stats, dev_stats)

    def save_checkpoint(self, save_dir: str, filename: str, epoch: int, train_stats: dict, dev_stats: Optional[dict]):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_stats": train_stats,
            "dev_stats": dev_stats,
            "history": self.history,
            "left_label2id": self.left_label2id,
            "right_label2id": self.right_label2id,
        }

        torch.save(ckpt, save_path / filename)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "history" in ckpt:
            self.history = ckpt["history"]

        return ckpt

    def save_config(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "model": getattr(self.model, "config", {}),
            "training": self.config,
        }

        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @torch.no_grad()
    def predict_batch(
        self,
        batch,
        id2token,
        id2left,
        id2right,
        left_threshold: float = 0.5,
        right_threshold: float = 0.5,
    ):
        
        self.model.eval()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            left_ids=batch["left_ids"],
            right_ids=batch["right_ids"],
            attention_mask=attention_mask,
        )

        left_logits = outputs["left_logits"]    # [B, T, L]
        right_logits = outputs["right_logits"]  # [B, T, R]

        pred_left = (torch.sigmoid(left_logits) > left_threshold)
        pred_right = (torch.sigmoid(right_logits) > right_threshold)

        # pred_left_ids_list = []
        # pred_right_ids_list = []
        batch_preds = []

        for i in range(pred_left.size(0)):
            valid_len = attention_mask[i].sum().item()

            # sent_left = []
            # sent_right = []
            sent_preds = []

            for j in range(valid_len):
                left_ids = torch.nonzero(pred_left[i, j], as_tuple=False).squeeze(1).tolist()
                right_ids = torch.nonzero(pred_right[i, j], as_tuple=False).squeeze(1).tolist()

                # sent_left.append(left_ids)
                # sent_right.append(right_ids)

                
                token  = {
                    "token": id2token[input_ids[i, j]], 
                    "left": [id2left[idx] for idx in left_ids],
                    "right": [id2right[idx] for idx in right_ids],
                    }
                sent_preds.append(token)
            batch_preds.append(sent_preds)
                    
                    

            # pred_left_ids_list.append(sent_left)
            # pred_right_ids_list.append(sent_right)

        # output = {
        #     "pred_left_ids": pred_left_ids_list,
        #     "pred_right_ids": pred_right_ids_list,
        # }


        # if id2left is not None:
        #     output["pred_left"] = [
        #         [[id2left[idx] for idx in ids] for ids in sent]
        #         for sent in pred_left_ids_list
        #     ]

        # if id2right is not None:
        #     output["pred_right"] = [
        #         [[id2right[idx] for idx in ids] for ids in sent]
        #         for sent in pred_right_ids_list
        #     ]

        return batch_preds

    @torch.no_grad()
    def predict_loader(
        self,
        dataloader,
        id2token,
        id2left,
        id2right,
        left_threshold: float = 0.5,
        right_threshold: float = 0.5,
        save_dir=None,
    ):
        if any(x is None for x in [id2token, id2left, id2right]):
            raise ValueError("id2token, id2left or id2right missing.")
        self.model.eval()

        all_pred_left_ids = []
        all_pred_right_ids = []
        all_pred_left = []
        all_pred_right = []

        all_preds = []

        for batch in dataloader:
            batch_preds = self.predict_batch(
                batch,
                id2token=id2token,
                id2left=id2left,
                id2right=id2right,
                left_threshold=left_threshold,
                right_threshold=right_threshold,
            )

            all_preds.extend(batch_preds)

            # all_pred_left_ids.extend(out["pred_left_ids"])
            # all_pred_right_ids.extend(out["pred_right_ids"])

            # if "pred_left" in out:
            #     all_pred_left.extend(out["pred_left"])
            # if "pred_right" in out:
            #     all_pred_right.extend(out["pred_right"])

        # result = {
        #     "pred_left_ids": all_pred_left_ids,
        #     "pred_right_ids": all_pred_right_ids,
        # }

        # if id2left is not None:
        #     result["pred_left"] = all_pred_left
        # if id2right is not None:
        #     result["pred_right"] = all_pred_right

        if save_dir is not None:
            with open(save_dir, "w", encoding="utf-8") as fout:
                json.dump(all_preds, fout, indent=2)
    
        return all_preds

    @torch.no_grad()
    def predict_tokens(
        self,
        tokens,
        left_ids,
        right_ids,
        token2id,
        id2left,
        id2right,
        left_threshold: float = 0.5,
        right_threshold: float = 0.5,
        # pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        if any(x is None for x in [id2left, id2right]):
            raise ValueError("id2left or id2right missing.")
        self.model.eval()

        unk_id = token2id[unk_token]
        input_ids = [token2id.get(tok, unk_id) for tok in tokens]

        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        outputs = self.model(
            input_ids=input_ids,
            left_ids=[left_ids],
            right_ids=[right_ids],
            attention_mask=attention_mask,
        )

        pred_left = (torch.sigmoid(outputs["left_logits"][0]) > left_threshold)
        pred_right = (torch.sigmoid(outputs["right_logits"][0]) > right_threshold)

        # pred_left_ids = []
        # pred_right_ids = []
        sent_pred = []

        for j in range(pred_left.size(0)):
            left_ids = torch.nonzero(pred_left[j], as_tuple=False).squeeze(1).tolist()
            right_ids = torch.nonzero(pred_right[j], as_tuple=False).squeeze(1).tolist()
            token  = {
                    "token": tokens[j], 
                    "left": [id2left[idx] for idx in left_ids],
                    "right": [id2right[idx] for idx in right_ids],
                    }
            sent_pred.append(token)
            

            # pred_left_ids.append(left_idx)
            # pred_right_ids.append(right_idx)

        # result = {
        #     "tokens": tokens,
        #     "pred_left_ids": pred_left_ids,
        #     "pred_right_ids": pred_right_ids,
        # }

        # if id2left is not None:
        #     result["pred_left"] = [[id2left[idx] for idx in ids] for ids in pred_left_ids]

        # if id2right is not None:
        #     result["pred_right"] = [[id2right[idx] for idx in ids] for ids in pred_right_ids]

        return sent_pred