import pytorch_lightning as pl
import argparse
import os

from TransformerColiee import TransformerColiee
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything

seed_everything(42)


def main(hparams):

    model = TransformerColiee(hparams)

    # ─────────────────────────────────────────────
    # Logger
    # ─────────────────────────────────────────────
    loggers = []

    if hparams.use_wandb:
        wandb_logger = WandbLogger(
            project="coliee-longformer",
            name=f"run-{hparams.run_name}",
        )
        wandb_logger.log_hyperparams(vars(hparams))
        loggers.append(wandb_logger)

    if hparams.use_tensorboard:
        tb_logger = TensorBoardLogger(
            "tb_logs",
            name="longformer-coliee",
            version=hparams.run_name,
        )
        loggers.append(tb_logger)

    # ─────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # ─────────────────────────────────────────────
    # Trainer (PL >= 2.x style)
    # ─────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus > 0 else "cpu",
        devices=hparams.gpus if hparams.gpus > 0 else 1,
        num_nodes=hparams.num_nodes,

        max_epochs=hparams.epochs,
        max_steps=hparams.num_training_steps,

        accumulate_grad_batches=hparams.trainer_batch_size,

        logger=loggers,
        callbacks=[checkpoint_callback],

        val_check_interval=hparams.val_check_interval,

        precision=16 if hparams.fp16 else 32,
    )

    trainer.fit(model)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Longformer COLIEE")

    # MODEL
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-5)

    # LOSS
    parser.add_argument("--loss_type", type=str, default="bce")  # bce | margin
    parser.add_argument("--pos_weight", type=float, default=1.0)

    # TRAINING
    parser.add_argument("--num_warmup_steps", type=int, default=2500)
    parser.add_argument("--num_training_steps", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_check_interval", type=int, default=2000)

    # DATA
    parser.add_argument("--queries_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--retrieval_path", type=str, default=None)

    # BATCH
    parser.add_argument("--train_bs", type=int, default=1)
    parser.add_argument("--eval_bs", type=int, default=1)
    parser.add_argument("--trainer_batch_size", type=int, default=8)

    # SYSTEM
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)

    # LOGGING
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--use_tensorboard", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="debug")

    # PRECISION
    parser.add_argument("--fp16", type=int, default=1)

    hparams = parser.parse_args()
    print(hparams)

    main(hparams)
