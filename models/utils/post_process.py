import numpy as np
import torch


def to_probas(activations):
    activations = torch.tensor(activations)
    if activations.shape[1] == 1: # binary class
        probas = torch.sigmoid(activations).numpy().flatten()
    else:
        probas = torch.softmax(activations, dim=-1).numpy()
    return probas

def to_labels(activations):
    if activations.shape[1] == 1: # binary class
        labels = (to_probas(activations) >= 0.5).astype(int)
    else:
        labels = np.argmax(activations, axis=-1)
    return labels