import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pytorch_lightning as pl
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from ColieeDataset import ColieeDataset


class TransformerColiee(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)  # 🔥 log config lên wandb

        self.loss_type = hparams.loss_type

        self.tokenizer = LongformerTokenizer.from_pretrained(hparams.model_name)

        self.model = LongformerForSequenceClassification.from_pretrained(
            hparams.model_name,
            num_labels=1
        )

        self.DatasetClass = ColieeDataset

    # ─────────────────────────────
    # Forward
    # ─────────────────────────────
    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        return outputs.logits.squeeze(-1)

    # ─────────────────────────────
    # Optimizer
    # ─────────────────────────────
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                self.hparams.num_warmup_steps,
                self.hparams.num_training_steps
            ),
            "interval": "step"
        }

        return [optimizer], [scheduler]

    # ─────────────────────────────
    # DataLoader
    # ─────────────────────────────
    @staticmethod
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "global_attention_mask": torch.stack([x["global_attention_mask"] for x in batch]),
            "label": torch.stack([x["label"] for x in batch]),
            "qid": [x["qid"] for x in batch],
        }

    def train_dataloader(self):
        dataset = self.DatasetClass(
            queries_path=self.hparams.queries_path,
            labels_path=self.hparams.labels_path,
            corpus_path=self.hparams.corpus_path,
            tokenizer=self.tokenizer,
            mode="train",
            max_seq_len=self.hparams.max_seq_len,
            retrieval_results_path=self.hparams.retrieval_path,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.train_bs,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        dataset = self.DatasetClass(
            queries_path=self.hparams.queries_path,
            labels_path=self.hparams.labels_path,
            corpus_path=self.hparams.corpus_path,
            tokenizer=self.tokenizer,
            mode="dev",
            max_seq_len=self.hparams.max_seq_len,
            retrieval_results_path=self.hparams.retrieval_path,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.eval_bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
        )

    # ─────────────────────────────
    # Training
    # ─────────────────────────────
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        labels = batch["label"].squeeze(1)

        logits = self.forward(input_ids, attention_mask, global_attention_mask)

        # ───── BCE ─────
        if self.loss_type == "bce":
            labels = labels.float()

            pos_weight = torch.tensor(
                [self.hparams.pos_weight],
                device=logits.device
            )

            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight
            )

        # ───── Margin ─────
        elif self.loss_type == "margin":
            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                loss = torch.tensor(0.0, device=logits.device)
            else:
                pos_scores = logits[pos_mask]
                neg_scores = logits[neg_mask]

                loss = torch.mean(
                    torch.clamp(1.0 - pos_scores.unsqueeze(1) + neg_scores.unsqueeze(0), min=0)
                )

        else:
            raise ValueError("loss_type must be 'bce' or 'margin'")

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # log lr
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True)

        return loss

    # ─────────────────────────────
    # Validation
    # ─────────────────────────────
    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        global_attention_mask = batch["global_attention_mask"]
        labels = batch["label"].squeeze(1)

        logits = self.forward(input_ids, attention_mask, global_attention_mask)

        # loss
        if self.loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
            probs = torch.sigmoid(logits)

        elif self.loss_type == "margin":
            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                return

            loss = torch.clamp(
                1.0 - logits[pos_mask].mean() + logits[neg_mask].mean(),
                min=0
            )
            probs = torch.sigmoid(logits)

        # log step loss (optional)
        self.log("val_step_loss", loss, prog_bar=False)

        self.val_outputs.append({
            "loss": loss.detach(),
            "probs": probs.detach(),
            "labels": labels.detach(),
            "qids": batch["qid"]
        })


    def on_validation_epoch_start(self):
        self.val_outputs = []

    def on_validation_epoch_end(self):

        if len(self.val_outputs) == 0:
            return

        outputs = self.val_outputs

        # ─── loss ───
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # ─── MRR ───
        mrr = self.compute_mrr(outputs)
        mrr10 = self.compute_mrr(outputs, k=10)

        # ─── log ───
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("mrr", mrr, prog_bar=True)
        self.log("mrr@10", mrr10)

        print(f"\nVAL LOSS: {avg_loss:.4f} | MRR: {mrr:.4f} | MRR@10: {mrr10:.4f}")

        # clear buffer
        self.val_outputs.clear()



    def compute_mrr(self, outputs, k=None):

        all_probs = []
        all_labels = []
        all_qids = []

        for x in outputs:
            all_probs.extend(x["probs"].cpu().tolist())
            all_labels.extend(x["labels"].cpu().tolist())
            all_qids.extend(x["qids"])

        df = pd.DataFrame({
            "qid": all_qids,
            "prob": all_probs,
            "label": all_labels
        })

        mrr = 0.0
        q_count = 0

        for qid in df.qid.unique():
            tmp = df[df.qid == qid].sort_values("prob", ascending=False)

            if k:
                tmp = tmp.head(k)

            relevant = tmp[tmp.label == 1]

            if len(relevant) == 0:
                continue

            tmp = tmp.reset_index(drop=True)
            relevant = tmp[tmp.label == 1]

            first_rank = relevant.index[0] + 1
            mrr += 1.0 / first_rank
            q_count += 1

        return mrr / max(q_count, 1)
