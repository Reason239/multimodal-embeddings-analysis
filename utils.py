import numpy as np
import torch


def centered(embedding_matrix):
    return embedding_matrix - embedding_matrix.mean(0)


def normed(embedding_matrix):
    if isinstance(embedding_matrix, torch.Tensor):
        return embedding_matrix / embedding_matrix.norm(dim=-1, keepdim=True)
    return embedding_matrix / np.linalg.norm(embedding_matrix, axis=-1, keepdims=True)
