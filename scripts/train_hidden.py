from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stegmark.data.dataset import HiddenImageDataset
from stegmark.training.trainer import HiddenTrainer, HiddenTrainerConfig


def read_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split(":", maxsplit=1)
        data[key.strip()] = value.strip()
    return data


def load_config(path: Path) -> HiddenTrainerConfig:
    raw = read_simple_yaml(path)
    return HiddenTrainerConfig(
        message_bits=int(raw.get("message_bits", "32")),
        image_size=int(raw.get("image_size", "128")),
        batch_size=int(raw.get("batch_size", "16")),
        learning_rate=float(raw.get("learning_rate", "0.001")),
        image_weight=float(raw.get("image_weight", "1.0")),
        message_weight=float(raw.get("message_weight", "5.0")),
        device=raw.get("device", "cpu"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the HiDDeN encoder/decoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--output_dir", type=Path, default=Path("weights/hidden"))
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = HiddenImageDataset(
        args.dataset,
        message_bits=config.message_bits,
        image_size=config.image_size,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    trainer = HiddenTrainer(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    no_improve_count = 0
    float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[dict] = []

        for batch_images, batch_messages in loader:
            metrics = trainer.train_step({"image": batch_images, "message": batch_messages})
            epoch_losses.append(metrics)

        # 计算当轮平均 loss
        keys = epoch_losses[0].keys() if epoch_losses else []
        avg: dict[str, float] = {}
        for k in keys:
            avg[k] = sum(m[k] for m in epoch_losses) / len(epoch_losses)

        if epoch % 10 == 0:
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"[epoch {epoch}/{args.epochs}]  {loss_str}")

        if epoch % 20 == 0:
            torch.save(
                trainer.encoder.state_dict(),
                args.output_dir / f"encoder_ep{epoch}.pt",
            )
            torch.save(
                trainer.decoder.state_dict(),
                args.output_dir / f"decoder_ep{epoch}.pt",
            )

        # 早停：decoder_loss 连续 5 epoch < 0.05
        decoder_loss = avg.get("decoder_loss", avg.get("message_loss", float("inf")))
        if decoder_loss < 0.05:
            no_improve_count += 1
            if no_improve_count >= 5:
                print(f"早停：decoder_loss 已连续 {no_improve_count} epoch < 0.05，停止训练。")
                break
        else:
            no_improve_count = 0

    torch.save(trainer.encoder.state_dict(), args.output_dir / "encoder_final.pt")
    torch.save(trainer.decoder.state_dict(), args.output_dir / "decoder_final.pt")
    print(f"训练完成，权重已保存至 {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
