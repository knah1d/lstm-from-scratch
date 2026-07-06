import numpy as np


def mse_loss(pred, target):
    diff = pred - target
    loss = np.mean(diff ** 2)
    grad = (2.0 / diff.size) * diff
    return loss, grad


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_entropy_loss(logits, labels):
    # logits: (batch, num_classes), labels: (batch,) int class indices
    batch = logits.shape[0]
    probs = softmax(logits)
    correct_probs = probs[np.arange(batch), labels]
    loss = -np.mean(np.log(correct_probs + 1e-12))

    grad = probs.copy()
    grad[np.arange(batch), labels] -= 1.0
    grad /= batch
    return loss, grad
