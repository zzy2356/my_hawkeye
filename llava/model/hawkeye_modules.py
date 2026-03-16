from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


CLASSES = [
    'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
    'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
    'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
    'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
    'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
    'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
    'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
    'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
    'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
    'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
    'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
    'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
    'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
    'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra',
]


class PoseTower(nn.Module):
    def __init__(self, hidden_size: int, pose_dim: int = 85):
        super().__init__()
        self.pose_dim = pose_dim
        self.proj = nn.Linear(pose_dim, hidden_size)

    def forward(self, pose_values: torch.Tensor) -> torch.Tensor:
        if pose_values.ndim == 1:
            pose_values = pose_values.unsqueeze(0)
        if pose_values.ndim > 2:
            pose_values = pose_values.reshape(pose_values.shape[0], -1)
        return self.proj(pose_values)


class GTNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_attr_dim: int):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.edge_attr_linear = nn.Linear(edge_attr_dim, in_channels)
        self.edge_attr_dim = edge_attr_dim

    def forward(self, x, edge_index, edge_attr=None):
        edge_index, edge_attr = self._add_self_loops_with_edge_attr(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=x.size(0),
            edge_attr_dim=self.edge_attr_dim,
            device=x.device,
            dtype=x.dtype,
        )
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.linear(x)

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return x_j
        return x_j + self.edge_attr_linear(edge_attr.to(dtype=x_j.dtype))

    @staticmethod
    def _add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes, edge_attr_dim, device, dtype):
        self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index.to(device=device), self_loops], dim=1)
        if edge_attr is None:
            return edge_index, None
        self_loop_attr = torch.zeros((num_nodes, edge_attr_dim), device=device, dtype=edge_attr.dtype)
        edge_attr = torch.cat([edge_attr.to(device=device), self_loop_attr], dim=0)
        return edge_index, edge_attr


class SceneGraphTower(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        node_dim: int = 151,
        edge_attr_dim: int = 51,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_attr_dim = edge_attr_dim
        self.conv_layers = nn.ModuleList()
        channels = node_dim
        for _ in range(num_layers):
            self.conv_layers.append(GTNLayer(channels, hidden_size, edge_attr_dim))
            channels = hidden_size
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, scene_feat: torch.Tensor) -> torch.Tensor:
        if scene_feat.ndim == 1:
            scene_feat = scene_feat.unsqueeze(0)
        if scene_feat.numel() == 0:
            return torch.zeros(1, self.hidden_size, device=scene_feat.device, dtype=scene_feat.dtype)

        probas = scene_feat[:, :51]
        probas_sub = scene_feat[:, 51:202]
        probas_obj = scene_feat[:, 202:]

        nodes = []
        node_features = []
        edges = []
        edge_features = []

        for i in range(probas.shape[0]):
            sub = CLASSES[probas_sub[i].argmax().item()]
            obj = CLASSES[probas_obj[i].argmax().item()]
            if sub not in nodes:
                nodes.append(sub)
                node_features.append(probas_sub[i])
            if obj not in nodes:
                nodes.append(obj)
                node_features.append(probas_obj[i])
            edges.append((sub, obj))
            edge_features.append(probas[i])

        if not node_features:
            return torch.zeros(1, self.hidden_size, device=scene_feat.device, dtype=scene_feat.dtype)

        edge_index = torch.tensor(
            [[nodes.index(src), nodes.index(dst)] for src, dst in edges],
            dtype=torch.long,
            device=scene_feat.device,
        ).t().contiguous()
        node_features = torch.stack(node_features, dim=0).to(device=scene_feat.device, dtype=scene_feat.dtype)
        edge_features = torch.stack(edge_features, dim=0).to(device=scene_feat.device, dtype=scene_feat.dtype)

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr)
            x = F.gelu(x)
        return self.output_proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ResamplerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class HawkeyeMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 2,
        num_layers: int = 1,
        num_heads: int = 8,
        scene_token_count: int = 30,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.scene_token_count = scene_token_count
        self.routers = Mlp(hidden_size, hidden_size * 4, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.ModuleList([ResamplerBlock(hidden_size, num_heads) for _ in range(num_layers)])
                for _ in range(num_experts)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.last_routing_weights: Optional[torch.Tensor] = None
        self.last_debug: Dict[str, Any] = {}

    def _resize_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == self.scene_token_count:
            return x
        if x.size(1) > self.scene_token_count:
            return x[:, :self.scene_token_count]
        pad = x[:, -1:, :].expand(-1, self.scene_token_count - x.size(1), -1)
        return torch.cat([x, pad], dim=1)

    def forward(self, pose_feat: torch.Tensor, scene_feat: torch.Tensor) -> torch.Tensor:
        if pose_feat.ndim == 2:
            pose_feat = pose_feat.unsqueeze(0)
        if scene_feat.ndim == 2:
            scene_feat = scene_feat.unsqueeze(0)
        fused = torch.cat((pose_feat, scene_feat), dim=1)
        # Resize fused sequence to scene_token_count BEFORE computing routing
        # weights so that routing_weights and expert_hidden share the same
        # token dimension and the broadcast in (expert_hidden * expert_weight)
        # does not raise a size mismatch.
        fused = self._resize_sequence(fused)
        # routing_weights: (B, scene_token_count, num_experts)
        routing_weights = torch.softmax(self.routers(fused), dim=-1)
        self.last_routing_weights = routing_weights.detach().float().cpu()
        expert_outputs = []
        for expert_id, blocks in enumerate(self.experts):
            expert_hidden = fused
            for block in blocks:
                expert_hidden = block(expert_hidden)
            # expert_hidden is already scene_token_count tokens; no resize needed.
            expert_weight = routing_weights[:, :, expert_id].unsqueeze(-1)  # (B, scene_token_count, 1)
            expert_outputs.append(expert_hidden * expert_weight)
        output = self.out_proj(sum(expert_outputs))
        dominant_experts = routing_weights.argmax(dim=-1)
        expert_token_counts = [
            int((dominant_experts == expert_id).sum().item()) for expert_id in range(self.num_experts)
        ]
        entropy = -(routing_weights * routing_weights.clamp_min(1e-8).log()).sum(dim=-1)
        preview_len = min(8, routing_weights.shape[1])
        self.last_debug = {
            "fused_shape": list(fused.shape),
            "routing_weights_shape": list(routing_weights.shape),
            "routing_weight_means": routing_weights.mean(dim=(0, 1)).detach().float().cpu().tolist(),
            "routing_weight_max": routing_weights.amax(dim=(0, 1)).detach().float().cpu().tolist(),
            "routing_entropy_mean": float(entropy.mean().detach().cpu().item()),
            "dominant_expert_counts": expert_token_counts,
            "routing_preview": routing_weights[0, :preview_len].detach().float().cpu().tolist(),
            "output_shape": list(output.shape),
        }
        return output


def build_pose_tower(hidden_size: int, pose_dim: int = 85) -> PoseTower:
    return PoseTower(hidden_size=hidden_size, pose_dim=pose_dim)


def build_pose_projector(hidden_size: int) -> nn.Module:
    return nn.Linear(hidden_size, hidden_size)


def build_scene_tower(hidden_size: int) -> SceneGraphTower:
    return SceneGraphTower(hidden_size=hidden_size)


def build_scene_projector(hidden_size: int) -> nn.Module:
    return nn.Linear(hidden_size, hidden_size)


def build_moe(hidden_size: int, scene_token_count: int = 30) -> HawkeyeMoE:
    num_heads = max(1, min(8, hidden_size // 256))
    return HawkeyeMoE(hidden_size=hidden_size, num_heads=num_heads, scene_token_count=scene_token_count)


def build_moe_projector(hidden_size: int) -> nn.Module:
    return nn.Linear(hidden_size, hidden_size)
