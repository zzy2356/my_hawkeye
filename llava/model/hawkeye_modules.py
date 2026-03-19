import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


CLASSES = [
    "N/A", "airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear", "bed", "bench", "bike",
    "bird", "board", "boat", "book", "boot", "bottle", "bowl", "box", "boy", "branch", "building",
    "bus", "cabinet", "cap", "car", "cat", "chair", "child", "clock", "coat", "counter", "cow", "cup",
    "curtain", "desk", "dog", "door", "drawer", "ear", "elephant", "engine", "eye", "face", "fence",
    "finger", "flag", "flower", "food", "fork", "fruit", "giraffe", "girl", "glass", "glove", "guy",
    "hair", "hand", "handle", "hat", "head", "helmet", "hill", "horse", "house", "jacket", "jean",
    "kid", "kite", "lady", "lamp", "laptop", "leaf", "leg", "letter", "light", "logo", "man", "men",
    "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange", "pant", "paper", "paw",
    "people", "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player", "pole", "post",
    "pot", "racket", "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf", "shirt",
    "shoe", "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier", "sneaker", "snow",
    "sock", "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire", "toilet", "towel",
    "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase", "vegetable", "vehicle",
    "wave", "wheel", "window", "windshield", "wing", "wire", "woman", "zebra",
]


class PoseTower(nn.Module):
    """Faithful to the original pose_feat contract: [T, 17, 5] -> [T, hidden]."""

    def __init__(self, hidden_size: int, pose_dim: int = 85):
        super().__init__()
        self.pose_projector = nn.Linear(pose_dim, hidden_size)

    def forward(self, pose_values: torch.Tensor) -> torch.Tensor:
        if pose_values.ndim == 1:
            pose_values = pose_values.unsqueeze(0)
        pose_values = pose_values.reshape(pose_values.shape[0], -1)
        return self.pose_projector(pose_values)


class GTNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_attr_dim: int):
        super().__init__(aggr="add")
        self.linear = nn.Linear(in_channels, out_channels)
        self.edge_attr_linear = nn.Linear(edge_attr_dim, in_channels)
        self.edge_attr_dim = edge_attr_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        edge_index, edge_attr = self.add_self_loops_with_edge_attr(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=x.size(0),
            edge_attr_dim=self.edge_attr_dim,
            device=x.device,
            dtype=x.dtype,
        )
        return self.linear(self.propagate(edge_index, x=x, edge_attr=edge_attr))

    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor]):
        if edge_attr is None:
            return x_j
        return x_j + self.edge_attr_linear(edge_attr.to(dtype=x_j.dtype))

    @staticmethod
    def add_self_loops_with_edge_attr(
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int,
        edge_attr_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index.to(device=device), self_loops], dim=1)
        if edge_attr is None:
            return edge_index, None
        self_loop_attr = torch.zeros((num_nodes, edge_attr_dim), device=device, dtype=edge_attr.dtype)
        edge_attr = torch.cat([edge_attr.to(device=device), self_loop_attr], dim=0)
        return edge_index, edge_attr


class SceneGraphTower(nn.Module):
    """Preserves the original GTN-style scene contract used by Hawkeye."""

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
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, scene_feat: torch.Tensor) -> torch.Tensor:
        if scene_feat.ndim == 1:
            scene_feat = scene_feat.unsqueeze(0)
        if scene_feat.numel() == 0:
            return torch.zeros(1, self.hidden_size, device=scene_feat.device, dtype=scene_feat.dtype)

        probas = scene_feat[:, :51]
        probas_sub = scene_feat[:, 51:202]
        probas_obj = scene_feat[:, 202:]
        nodes = []
        edges = []
        node_features = []
        edge_features = probas

        for idx in range(probas.shape[0]):
            sub = CLASSES[probas_sub[idx].argmax().item()]
            obj = CLASSES[probas_obj[idx].argmax().item()]
            if sub not in nodes:
                nodes.append(sub)
                node_features.append(probas_sub[idx])
            if obj not in nodes:
                nodes.append(obj)
                node_features.append(probas_obj[idx])
            edges.append((sub, obj))

        if not node_features:
            return torch.zeros(1, self.hidden_size, device=scene_feat.device, dtype=scene_feat.dtype)

        edge_index = torch.tensor(
            [[nodes.index(src), nodes.index(dst)] for src, dst in edges],
            dtype=torch.long,
            device=scene_feat.device,
        ).t().contiguous()
        node_features = torch.stack(node_features, dim=0).to(device=scene_feat.device, dtype=scene_feat.dtype)
        edge_features = edge_features.to(device=scene_feat.device, dtype=scene_feat.dtype)

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr)

        # Preserve the original contract exactly. With unique batch ids this keeps
        # one output row per graph node while still matching the legacy code path.
        x = gnn.global_mean_pool(x, torch.arange(0, x.size(0), dtype=torch.long, device=x.device))
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


@dataclass
class ModelArgs:
    dim: int
    n_layers: int = 8
    n_heads: int = 16
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_batch_size: int = 8
    max_seq_len: int = 256


class Attention(nn.Module):
    """Original Hawkeye-style resampler attention with a flash-attn fallback."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del start_pos, freqs_cis, prompt
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if flash_attn_func is not None:
            output = flash_attn_func(xq, xk, xv, dropout_p=0.0, causal=mask is not None)
            return self.wo(output.contiguous().view(bsz, seqlen, -1))

        q = xq.permute(0, 2, 1, 3)
        k = xk.permute(0, 2, 1, 3)
        v = xv.permute(0, 2, 1, 3)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        output = output.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)
        return h + self.feed_forward(self.ffn_norm(h))


class Mlp(nn.Module):
    """Exact contract from the original Hawkeye router MLP."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class HawkeyeMoE(nn.Module):
    """Faithful to the original Hawkeye MOE contract, with debug summaries."""

    def __init__(self, hidden_size: int, num_experts: int = 2, num_resample_layers: int = 1, max_route_tokens: int = 30):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_resample_layers = num_resample_layers
        self.max_route_tokens = max_route_tokens

        params = ModelArgs(dim=hidden_size)
        self.resample_layers = nn.ModuleDict()
        for expert_id in range(num_experts):
            expert_key = str(expert_id)
            self.resample_layers[expert_key] = nn.ModuleList()
            resampler_params = copy.deepcopy(params)
            for layer_id in range(num_resample_layers):
                self.resample_layers[expert_key].append(TransformerBlock(layer_id, resampler_params))

        self.resample_tokens = nn.ParameterDict()
        self.routers = nn.ModuleDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        self.start_tag = nn.ParameterDict()
        self.end_tag = nn.ParameterDict()

        for modal in ("pose",):
            self.routers[modal] = Mlp(hidden_size, hidden_size * 4, num_experts)
            self.resample_tokens[modal] = nn.Parameter(torch.empty([1, max_route_tokens, hidden_size]))
            nn.init.normal_(self.resample_tokens[modal], std=0.02)
            self.clip_proj1[modal] = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
            self.clip_proj2[modal] = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
            self.start_tag[modal] = nn.Parameter(torch.rand(1, 1, hidden_size))
            self.end_tag[modal] = nn.Parameter(torch.rand(1, 1, hidden_size))

        self.last_routing_weights: Optional[torch.Tensor] = None
        self.last_debug: Dict[str, Any] = {}

    def forward(self, pose_feat: torch.Tensor, scene_feat: torch.Tensor) -> torch.Tensor:
        if pose_feat.ndim == 2:
            pose_feat = pose_feat.unsqueeze(0)
        if scene_feat.ndim == 2:
            scene_feat = scene_feat.unsqueeze(0)

        image_feats = torch.cat((pose_feat, scene_feat), dim=1)
        routing_weights = self.routers["pose"](image_feats).sigmoid()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        self.last_routing_weights = routing_weights.detach().float().cpu()

        image_feats_experts = []
        for expert_id in range(self.num_experts):
            image_feats_expert = image_feats
            for layer in self.resample_layers[str(expert_id)]:
                image_feats_expert = layer(image_feats_expert, 0, None, None)

            keep_len = min(image_feats_expert.size(1), self.resample_tokens["pose"].size(1))
            image_feats_expert = image_feats_expert[:, :keep_len]
            routing_weight = routing_weights[:, :keep_len, expert_id]
            image_feats_expert = image_feats_expert * routing_weight[:, :, None]
            image_feats_experts.append(image_feats_expert)

        image_feats = sum(image_feats_experts)
        image_feats = self.clip_proj2["pose"](image_feats)
        dominant_experts = routing_weights.argmax(dim=-1)
        expert_token_counts = [
            int((dominant_experts == expert_id).sum().item()) for expert_id in range(self.num_experts)
        ]
        entropy = -(routing_weights * routing_weights.clamp_min(1e-8).log()).sum(dim=-1)
        preview_len = min(8, routing_weights.shape[1])
        self.last_debug = {
            "fused_shape": list(torch.cat((pose_feat, scene_feat), dim=1).shape),
            "routing_weights_shape": list(routing_weights.shape),
            "routing_weight_means": routing_weights.mean(dim=(0, 1)).detach().float().cpu().tolist(),
            "routing_weight_max": routing_weights.amax(dim=(0, 1)).detach().float().cpu().tolist(),
            "routing_entropy_mean": float(entropy.mean().detach().cpu().item()),
            "dominant_expert_counts": expert_token_counts,
            "routing_preview": routing_weights[0, :preview_len].detach().float().cpu().tolist(),
            "output_shape": list(image_feats.shape),
        }
        return image_feats.reshape(-1, self.hidden_size)


def build_pose_tower(hidden_size: int, pose_dim: int = 85) -> PoseTower:
    return PoseTower(hidden_size=hidden_size, pose_dim=pose_dim)


def build_pose_projector(hidden_size: int) -> nn.Module:
    return nn.Linear(hidden_size, hidden_size)


def build_scene_tower(hidden_size: int) -> SceneGraphTower:
    return SceneGraphTower(hidden_size=hidden_size)


def build_scene_projector(hidden_size: int) -> nn.Module:
    return nn.Linear(hidden_size, hidden_size)


def build_moe(hidden_size: int, scene_token_count: int = 30) -> HawkeyeMoE:
    return HawkeyeMoE(hidden_size=hidden_size, max_route_tokens=scene_token_count)


def build_moe_projector(hidden_size: int) -> nn.Module:
    return nn.Linear(hidden_size, hidden_size)
