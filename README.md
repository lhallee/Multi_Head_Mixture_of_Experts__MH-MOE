# Unofficial Multi-Head Mixture-of-Experts (MH-MOE) open-source implementation

The token-wise multihead routing mechanism suggested in [Multi-Head Mixture-of-Experts](https://arxiv.org/pdf/2404.15045) implemented in PyTorch. Tried to fill in all the paper's details correctly in the most efficient way, please create an issue or pull request with any suggestions. Looks like the official implementation will eventually be released [here](https://github.com/yushuiwx/MH-MoE).

## Usage

```
git clone https://github.com/lhallee/Multi_Head_Mixture_of_Experts__MH-MOE.git
```

```
from mhmoe import MultiHeadMoeBlock, Expert

# config = your_config
# requires hidden_size, num_experts, num_heads, topk, intermediate_size, hidden_dropout_prob

moe_block = MultiHeadMoeBlock(config, Expert)

x = torch.rand(batch_size, seq_len, hidden_size)
x = moe_block(x)
x.shape # (batch_size, seq_len, hidden_size)
```


### Test results: batch size 4 sequence length 128 on 3070

Config 0: 100%|███████████████████████████████| 100/100 [00:00<00:00, 334.34it/s]

Configuration: {'hidden_size': 768, 'num_experts': 8, 'num_heads': 12, 'topk': 2, 'intermediate_size': 3072, 'hidden_dropout_prob': 0.1}

Average execution time: 0.0030 seconds

Config 1: 100%|███████████████████████████████| 100/100 [00:00<00:00, 145.05it/s]

Configuration: {'hidden_size': 1024, 'num_experts': 16, 'num_heads': 16, 'topk': 4, 'intermediate_size': 4096, 'hidden_dropout_prob': 0.1}       

Average execution time: 0.0069 seconds

Config 2: 100%|███████████████████████████████| 100/100 [00:01<00:00, 50.06it/s]

Configuration: {'hidden_size': 1536, 'num_experts': 32, 'num_heads': 24, 'topk': 8, 'intermediate_size': 6144, 'hidden_dropout_prob': 0.1}

Average execution time: 0.0200 seconds
