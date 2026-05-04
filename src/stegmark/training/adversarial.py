from __future__ import annotations

import torch
import torch.nn as nn


def adversarial_perturbation(encoded, decoder, original_message, *, epsilon=0.03, alpha=0.005, steps=5):
    """PGD attack: generate adversarial perturbation to maximize decoder extraction error."""
    delta = torch.zeros_like(encoded, requires_grad=True)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        logits = decoder(encoded + delta)
        loss = criterion(logits, original_message)
        grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
        delta = delta + alpha * torch.sign(grad)
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = delta.detach().requires_grad_(True)
    return delta.detach()


def adversarial_training_step(
    encoder, decoder, cover, message,
    *, epsilon=0.03, adv_steps=5, image_loss_fn, message_loss_fn, lambda_adv=0.3,
):
    """Single adversarial training step: normal embed + PGD attack + joint decoder training."""
    encoded = encoder(cover, message)
    delta = adversarial_perturbation(encoded, decoder, message, epsilon=epsilon, steps=adv_steps)
    attacked = encoded + delta
    img_loss = image_loss_fn(encoded, cover)
    adv_msg_loss = message_loss_fn(decoder(attacked), message)
    return {
        "image_loss": float(img_loss.item()),
        "adv_message_loss": float(adv_msg_loss.item()),
        "total_loss": float((img_loss + lambda_adv * adv_msg_loss).item()),
    }


__all__ = ["adversarial_perturbation", "adversarial_training_step"]
