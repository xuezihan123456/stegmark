from __future__ import annotations

import importlib

import pytest


def test_hidden_encoder_output_shape() -> None:
    torch = pytest.importorskip("torch")

    from stegmark.nn.hidden.encoder import HiddenEncoder

    encoder = HiddenEncoder(message_bits=32)
    image = torch.rand(2, 3, 128, 128)
    message = torch.randint(0, 2, (2, 32), dtype=torch.float32)

    encoded = encoder(image, message)

    assert encoded.shape == image.shape


def test_hidden_decoder_output_shape() -> None:
    torch = pytest.importorskip("torch")

    from stegmark.nn.hidden.decoder import HiddenDecoder

    decoder = HiddenDecoder(message_bits=32)
    image = torch.rand(2, 3, 128, 128)

    logits = decoder(image)

    assert logits.shape == (2, 32)


def test_hidden_discriminator_output_shape() -> None:
    torch = pytest.importorskip("torch")

    from stegmark.nn.hidden.discriminator import HiddenDiscriminator

    discriminator = HiddenDiscriminator()
    image = torch.rand(2, 3, 128, 128)

    scores = discriminator(image)

    assert scores.shape == (2, 1)


def test_hidden_conv_block_factory_is_shared() -> None:
    encoder_module = importlib.import_module("stegmark.nn.hidden.encoder")
    decoder_module = importlib.import_module("stegmark.nn.hidden.decoder")
    discriminator_module = importlib.import_module("stegmark.nn.hidden.discriminator")
    layers_module = importlib.import_module("stegmark.nn.hidden.layers")

    assert encoder_module.conv_block is layers_module.conv_block
    assert decoder_module.conv_block is layers_module.conv_block
    assert discriminator_module.conv_block is layers_module.conv_block
    assert not hasattr(encoder_module, "_conv_block")
    assert not hasattr(decoder_module, "_conv_block")
    assert not hasattr(discriminator_module, "_conv_block")
