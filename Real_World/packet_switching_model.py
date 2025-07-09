import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class PacketSwitchingPINN(nn.Module):
    def __init__(self, input_dim, n_experts=6, expert_hidden=64):
        super().__init__()
        self.input_dim = input_dim
        self.n_experts = n_experts
        
        # Router networks
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim // 8, 32),
                nn.ReLU(),
                nn.Linear(32, n_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(8)
        ])
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim // 8, expert_hidden),
                nn.Tanh(),
                nn.Linear(expert_hidden, expert_hidden // 2),
                nn.Tanh(),
                nn.Linear(expert_hidden // 2, 3)  # u, v, p
            ) for _ in range(n_experts)
        ])
        
        # Aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        block_size = self.input_dim // 8
        
        outputs = []
        routing_weights = []
        
        for i in range(8):
            # Extract block
            block = x[:, i*block_size:(i+1)*block_size]
            
            # Get routing weights
            routing = self.routers[i](block)
            routing_weights.append(routing)
            
            # Expert processing
            expert_outputs = torch.stack([
                expert(block) for expert in self.experts
            ], dim=1)  # [batch, n_experts, 3]
            
            # Weighted combination
            weighted_output = (routing.unsqueeze(-1) * expert_outputs).sum(dim=1)
            outputs.append(weighted_output)
        
        # Aggregate across blocks
        final_output = torch.stack(outputs).mean(dim=0)
        return self.aggregator(final_output)
