from __future__ import annotations

import numpy as np

from stegmark.evaluation.attacks import apply_attack
from stegmark.evaluation.benchmark import _bit_accuracy


def test_bit_accuracy_penalizes_missing_bits() -> None:
    assert _bit_accuracy((1, 1, 1, 1), (1, 1)) == 0.5


def test_bit_accuracy_penalizes_extra_bits() -> None:
    assert _bit_accuracy((1, 1), (1, 1, 0, 0)) == 0.5


def test_gaussian_noise_seed_changes_attack_output() -> None:
    image = np.full((32, 32, 3), 127, dtype=np.uint8)

    first = apply_attack(image, "gaussian_noise_0.03", seed=1)
    second = apply_attack(image, "gaussian_noise_0.03", seed=2)

    assert not np.array_equal(first, second)
