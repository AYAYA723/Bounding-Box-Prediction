import torch
from torch import nn
from torch.nn import functional as F

from predict_corner.supervision import spatial_soft_argmax


def position_embedding(grid_size, hidden_dim):
    H, W = grid_size
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    frequency = 10000 ** (-torch.arange(hidden_dim // 4).float() / (hidden_dim // 4))
    x = x.flatten()[:, None] * frequency
    y = y.flatten()[:, None] * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)[None]


def unpatchify(patches, grid_size, patch_size, num_corners):
    N = patches.shape[0]
    H, W = grid_size
    patches = patches.reshape(N, H, W, patch_size, patch_size, num_corners)  # Restore patch grids.
    return patches.permute(0, 5, 1, 3, 2, 4).reshape(
        N, num_corners, H * patch_size, W * patch_size)


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, tokens):
        N, L, D = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(N, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # (N, heads, L, head_dim)
        attention = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        attention = attention.transpose(1, 2).reshape(N, L, D)
        tokens = tokens + self.projection(attention)
        return tokens + self.mlp(self.norm2(tokens))


class HeatmapDecoder(nn.Module):
    def __init__(self, image_size=(480, 640), patch_size=14, input_dim=768,
                 hidden_dim=768, num_layers=12, num_heads=8, num_corners=8,
                 decoder_temperature=.25):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_corners = num_corners
        self.decoder_temperature = decoder_temperature
        H, W = image_size
        self.grid_size = ((H + patch_size - 1) // patch_size, (W + patch_size - 1) // patch_size)
        self.input_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(.1)
        )
        self.input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.query_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.register_buffer('position_embedding', position_embedding(self.grid_size, hidden_dim))
        self.transformer = nn.Sequential(*[
            SelfAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.heatmap_projection = nn.Linear(hidden_dim, patch_size * patch_size * num_corners)
        nn.init.normal_(self.heatmap_projection.weight, std=.001)
        nn.init.zeros_(self.heatmap_projection.bias)

    def forward(self, features):
        tokens = self.input_norm(self.input_transform(features))
        tokens = tokens + self.query_embedding.to(tokens.dtype) + self.position_embedding.to(tokens.dtype)
        tokens = self.transformer(tokens)  # (N, 1610, 768)
        patches = self.heatmap_projection(self.output_norm(tokens))  # (N, 1610, 14*14*8)
        logits = unpatchify(patches, self.grid_size, self.patch_size, self.num_corners)
        H, W = self.image_size
        logits = logits[:, :, :H, :W]  # Remove padded pixels.
        return logits.sigmoid(), spatial_soft_argmax(logits, self.decoder_temperature)
