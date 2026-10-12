import torch
import torch.nn as nn
import torch.nn.functional as F

BONES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)


def normalized_adjacency(joints: int = 17) -> torch.Tensor:
    matrix = torch.eye(joints, dtype=torch.float32)
    for parent, child in BONES:
        matrix[parent, child] = 1.0
        matrix[child, parent] = 1.0
    degree = matrix.sum(-1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * matrix * inv_sqrt[None, :]






class KinematicGraphBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_projection = nn.Linear(hidden, hidden)
        self.graph_projection = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        graph = torch.einsum("jk,bpkh->bpjh", adjacency, hidden)
        update = F.gelu(self.norm(
            self.self_projection(hidden) + self.graph_projection(graph)
        ))
        return hidden + update


class KinematicGraphRefiner(nn.Module):
    """Zero-initialized pose-level graph residual with pelvis recentering."""

    def __init__(self, hidden: int = 64, blocks: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(3, hidden)
        self.blocks = nn.ModuleList([KinematicGraphBlock(hidden) for _ in range(blocks)])
        self.output_projection = nn.Linear(hidden, 3)
        self.register_buffer("adjacency", normalized_adjacency())
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pose.ndim != 4 or pose.shape[-2:] != (17, 3):
            raise ValueError("pose must have shape (B,P,17,3)")
        centered = pose - pose[:, :, 0:1]
        hidden = self.input_projection(centered)
        for block in self.blocks:
            hidden = block(hidden, self.adjacency)
        delta = self.output_projection(hidden)
        refined = centered + delta
        refined = refined - refined[:, :, 0:1]
        return refined, refined - centered
