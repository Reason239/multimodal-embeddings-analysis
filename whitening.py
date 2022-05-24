import torch
import torch.nn as nn


class Whitener(nn.Module):
    def __init__(self, embedding_dim, k=None):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)
        self.W = torch.nn.Parameter(torch.eye(embedding_dim), requires_grad=False)
        self.k = k

    @torch.no_grad()
    def compute_parameters(self, embedding_matrix):
        self.mu.data = torch.mean(embedding_matrix, dim=0)
        cov = torch.cov(embedding_matrix.T)
        u, s, vh = torch.linalg.svd(cov)
        self.W.data = u / torch.sqrt(s)

    def forward(self, x):
        assert self.k is not None, 'You forgot to set k (dimensionality of the whitened embeddings)'
        return (x - self.mu) @ self.W[:, :self.k]


class CLIPWithWhitening(nn.Module):
    """CLIP model with whitening on top"""

    def __init__(self, clip_model, whitener):
        super().__init__()
        self.clip_model = clip_model
        self.whitener = whitener

    def encode_text(self, text):
        return self.whitener(self.clip_model.encode_text(text))

    def encode_image(self, image):
        return self.whitener(self.clip_model.emcode_image(image))

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
