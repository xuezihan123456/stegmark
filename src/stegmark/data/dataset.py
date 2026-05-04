from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class HiddenImageDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, root: Path | str, *, message_bits: int, image_size: int = 128) -> None:
        self.root = Path(root)
        self.message_bits = message_bits
        self.image_size = image_size
        self.image_paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not self.image_paths:
            raise ValueError(f"no training images found under {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as opened:
            resized = opened.convert("RGB").resize((self.image_size, self.image_size))
            array = np.asarray(resized, dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        generator = torch.Generator().manual_seed(index)
        message = torch.randint(
            0,
            2,
            (self.message_bits,),
            generator=generator,
            dtype=torch.int64,
        ).to(dtype=torch.float32)
        return image, message

