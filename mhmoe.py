import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    SwiGLU MLP
    """
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w3 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        # uncomment for it to initially be equivalent to two layer mlp
        # self.w2.weight.data.zero_()
        # self.w2.bias.data.fill_(1.0)
    
    def forward(self, hidden_states):
        hidden_states = F.silu(self.w1(hidden_states)) * self.w2(hidden_states)
        hidden_states = self.layer_norm(self.w3(hidden_states))
        return hidden_states


class MHRouter(nn.Module):
    def __init__(self, num_experts, hidden_dim, num_heads):
        super().__init__()
        self.expert_embedding = nn.Parameter(torch.randn(hidden_dim // num_heads, num_experts)) # (h, e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : hidden_states (B * L * n, h)
        return torch.matmul(x, self.expert_embedding) # (B * L * n, e)


class MultiHeadMoeBlock(nn.Module):
    def __init__(self, config, expert):
        super().__init__()
        self.hidden_dim = config.hidden_size # d
        self.num_experts = config.num_experts # e
        self.num_heads = config.num_heads # n
        self.topk = config.topk # k
        self.head_dim = self.hidden_dim // self.num_heads # h
        self.rounded_dim = (self.hidden_dim // self.num_heads) * self.num_heads # r

        self.multi_head_layer = nn.Linear(self.hidden_dim, self.rounded_dim)
        self.router = MHRouter(self.num_experts, self.hidden_dim, self.num_heads)

        config.hidden_size = self.head_dim
        config.intermediate_size = config.intermediate_size // self.num_heads
        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])
        self.merge_layer = nn.Linear(self.rounded_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : hidden_states (B, L, d)
        bs, L, _ = x.size()

        # If hidden_dim is not divisible by num_heads r != d
        x = self.multi_head_layer(x) # (B, L, r)
        x = x.reshape(bs * L * self.num_heads, self.head_dim).contiguous() # (B * L * n, h)

        ### Router
        router_logits = self.router(x) # (B * L * n, e)
        router_weights = router_logits.softmax(dim=-1) # (B * L * n, e)
        router_weights, selected_experts = torch.topk(router_weights, self.topk, dim=-1) # (B * L * n, k), (B * L * n, k)
        
        # Call experts densely, faster than selective loops
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # (B * L * n, e, h)
        # Select top-k expert outputs
        selected_expert_outputs = expert_outputs[torch.arange(expert_outputs.size(0)).unsqueeze(1), selected_experts] # (B * L * n, k, h)
        # Multiply selected expert outputs with router weights elementwise
        weighted_expert_outputs = selected_expert_outputs * router_weights.unsqueeze(-1) # (B * L * n, k, h)
        # Combine topk expert outputs
        x = weighted_expert_outputs.sum(dim=1) # (B * L * n, h)
        
        # Back to original shape
        x = x.reshape(bs, L, self.rounded_dim) # (B, L, r)
        x = self.merge_layer(x) # (B, L, d)
        return x
