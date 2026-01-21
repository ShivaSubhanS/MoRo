"""
Inspired from https://github.com/samsad35/VQ-MAE-S-code/blob/main/vqmae/vqmae/model/speech/Vector_Quantizer_EMA.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, cfg):
        """Initialize a vector quantizer

        Args:
            num_embeddings (int): Number of embeddings in the dictionary.
            embedding_dim (int): Dimension of each embedding in the dictionary.
            commitment_cost (float): Weight of the commitment loss for the VQ-VAE.
            decay (_type_): Decay for the moving averages used to train the codebook.
            epsilon (_type_, optional): Used for numerical stability. Defaults to 1e-5.
        """
        super(VectorQuantizerEMA, self).__init__()
        self.cfg = cfg

        self._embedding_dim = cfg.embedding_dim
        self._num_embeddings = cfg.num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        # self._embedding.requires_grad_(False)

        self.register_buffer("_ema_cluster_size", torch.zeros(self._num_embeddings))
        # self.register_buffer("_ema_cluster_size", torch.ones(self._num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(self._num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self.register_buffer("code_count", torch.ones(self._num_embeddings))

        self._decay = cfg.decay
        self._epsilon = 1e-5

    def _tile(self, x):
        nb_code, code_dim = x.shape
        if nb_code < self._num_embeddings:
            n_repeats = (self._num_embeddings + nb_code - 1) // nb_code
            std = 0.01 / torch.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out[:self._num_embeddings]

    def forward(
        self,
        inputs,
        detailed=False,
        return_indices=False,
    ):
        """Forward the quantization.

        Args:
            inputs (torch.Tensor): The continuous latent representation.
            detailed (bool, optional): Not used for a single quantizer but will be useful for RQ-VAEs. Defaults to False.
            return_indices (bool, optional): If True, returns the quantized latent representation and the corresponding indices. Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: loss is the loss associated with the encoding-decoding process, x_recon are the reconstructed meshes,
            and perplexity gives information about the codebook usage.
        """
        # convert inputs from BCL -> BLC
        inputs = inputs.contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices_return = (
            encoding_indices.clone().detach().view(input_shape[0], -1)
        )
        encoding_indices = encoding_indices.unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

            # code resetting
            self.code_count = self._decay * self.code_count + (
                1 - self._decay
            ) * torch.sum(encodings, 0)
            usage = (self.code_count.view(self._num_embeddings, 1) >= 1.0).float()
            self._embedding.weight = nn.Parameter(
                usage * self._embedding.weight
                + (1 - usage) * self._tile(flat_input) 
            )

        # Loss
        commitment_loss = torch.tensor(0.0).to(inputs)
        ortho_loss = torch.tensor(0.0).to(inputs)
        entropy_loss = torch.tensor(0.0).to(inputs)

        if True:
        # if self.cfg.loss_commitment > 0:
            commitment_loss = F.mse_loss(quantized.detach(), inputs)

        # orthogonal loss
        if True:
        # if self.cfg.loss_ortho > 0:
            used_embedding = self._embedding.weight[self.code_count >= 1.0]
            n_used = used_embedding.shape[0]
            normed_embedding = F.normalize(used_embedding, dim=1, p=2)
            cosine_similarity = torch.matmul(normed_embedding, normed_embedding.t())
            ortho_loss = (cosine_similarity**2).sum() / (
                n_used**2
            ) - 1 / n_used

        # quantization entropy loss
        if True:
        # if self.cfg.loss_entropy > 0:
            temperature = self.cfg.temperature 
            distances = (
                torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(self._embedding.weight**2, dim=1)
                - 2 * torch.matmul(flat_input, self._embedding.weight.t())
            )
            prob = -distances / temperature
            prob = F.softmax(prob, dim=1)
            entropy_loss = -torch.sum(prob * torch.log(prob + 1e-10), dim=1).mean()
            entropy_loss = entropy_loss / torch.log2(torch.tensor(self._num_embeddings))

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        coverage = (self.code_count >= 1.0).float().sum() / self._num_embeddings

        ret = {
            "commitment_loss": commitment_loss,
            "ortho_loss": ortho_loss,
            "entropy_loss": entropy_loss,
            "perplexity": perplexity,
            "coverage": coverage,
            "quantized": quantized.contiguous(),
            "indices": encoding_indices_return,
        }
        return ret

    def get_codebook_indices(self, input):
        """Get indices from the continuous latent representation.

        Args:
            input (torch.Tensor): The continuous latent representation.

        Returns:
            torch.Tensor: The quantized latent representation.
        """
        inputs = input.contiguous()
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight.to(flat_input) ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.to(flat_input).t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices.view(inputs.shape[0], -1)

    def quantify(self, encoding_indices):
        """Get quantized latent representation from the indices.

        Args:
            encoding_indices (torch.Tensor): The indices of the embeddings to select in the dictionary.

        Returns:
            torch.Tensor: The quantized latent representation.
        """
        mesh_embeds = self._embedding(encoding_indices)
        return mesh_embeds

    def get_codebook(self):
        """Get the codebook of the VQ-VAE.

        Returns:
            torch.Tensor: The codebook.
        """
        return self._embedding.weight


class RecursiveVectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_quantizers,
        num_embeddings,
        embedding_dim,
        decay,
        epsilon=1e-5,
        shared_codebook=False,
    ):
        """Initialize a RQ-VAE.

        Args:
            num_quantizers (int): Number of quantizations in the latent space.
            num_embeddings (int): Number of embeddings in each dictionary (is the same for all dictionaries).
            embedding_dim (int): Dimension of embeddings.
            commitment_cost (float): Weight of the commitment loss.
            decay (float): Decay for the moving averages.
            epsilon (float, optional): Used for numerical stability. Defaults to 1e-5.
            shared_codebook (bool, optional): If True, use the same codebook for all quantizers. Defaults to False.
        """
        super(RecursiveVectorQuantizerEMA, self).__init__()

        self.num_quantizers = num_quantizers
        self._embedding_dim = embedding_dim
        if shared_codebook:
            codebook = VectorQuantizerEMA(num_embeddings, embedding_dim, decay, epsilon)
            self.layers = nn.ModuleList([codebook for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList(
                [
                    VectorQuantizerEMA(num_embeddings, embedding_dim, decay, epsilon)
                    for _ in range(num_quantizers)
                ]
            )

    def forward(self, inputs, detailed=False):
        """Forward the succesive quantizations.

        Args:
            inputs (torch.Tensor): The continuous latent representation.
            detailed (bool, optional): If detailed, outputs all intermediate latent representations. Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: loss is the loss associated with the encoding-decoding process, x_recon are the reconstructed meshes,
            and perplexity gives information about the codebook usage.
        """
        quantized_out = torch.zeros_like(inputs)
        residuals = inputs

        all_losses = []
        all_perplexities = []
        all_encodings = []
        all_quantized = []

        for layer in self.layers:
            loss, quantized, perplexity, encodings = layer(residuals)
            residuals = residuals - quantized.detach()
            quantized_out = quantized_out + quantized

            all_losses.append(loss)
            all_perplexities.append(perplexity)
            all_encodings.append(encodings)
            if detailed:
                all_quantized.append(quantized_out)

        if detailed:
            return all_losses, all_quantized, all_perplexities, all_encodings
        return (
            sum(all_losses),
            quantized_out.contiguous(),
            sum(all_perplexities),
            all_encodings,
        )

    def quantify(self, encoding_indices):
        """Get quantized latent representation from the indices.

        Args:
            encoding_indices (torch.Tensor): The indices of the embeddings to select in the dictionary.

        Returns:
            torch.Tensor: The quantized latent representation.
        """
        all_mesh_embeds = []

        for i, layer in enumerate(self.layers):
            mesh_embeds = layer._embedding(encoding_indices[:, :, i])
            all_mesh_embeds.append(mesh_embeds.unsqueeze(2))
        return torch.cat(all_mesh_embeds, dim=2)

    def get_codebook_indices(self, inputs):
        """Get indices from the continuous latent representation.

        Args:
            input (torch.Tensor): The continuous latent representation.

        Returns:
            torch.Tensor: The quantized latent representation.
        """
        residuals = inputs

        all_indices = []

        for layer in self.layers:
            quantized, indices = layer(residuals, return_indices=True)
            residuals = residuals - quantized.detach()
            all_indices.append(indices.unsqueeze(-1))

        return torch.cat(all_indices, dim=-1)
