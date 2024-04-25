import torch
import time
from tqdm.auto import tqdm
from mhmoe import *


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = [
        {'hidden_size': 768,
         'num_experts': 8,
         'num_heads': 12,
         'topk': 2,
         'intermediate_size': 3072,
         'hidden_dropout_prob':0.1},
        {'hidden_size': 1024,
         'num_experts': 16,
         'num_heads': 16,
         'topk': 4,
         'intermediate_size': 4096,
         'hidden_dropout_prob':0.1},
        {'hidden_size': 1536,
         'num_experts': 32,
         'num_heads': 24,
         'topk': 8,
         'intermediate_size': 6144,
         'hidden_dropout_prob':0.1},
    ]

    # Benchmark the MultiHeadMoeBlock
    for i, config in enumerate(configs):
        moe_block = MultiHeadMoeBlock(type("Config", (object,), config), Expert).to(device)
        
        batch_size = 4
        seq_length = 128
        x = torch.randn(batch_size, seq_length, config['hidden_size']).to(device)
        
        # Warm-up run
        _ = moe_block(x)
        
        # Measure the execution time
        num_runs = 100
        start_time = time.time()
        for _ in tqdm(range(num_runs), desc=f'Config {i}'):
            _ = moe_block(x)
        end_time = time.time()
        
        torch.cuda.empty_cache()
        # Calculate the average execution time
        avg_time = (end_time - start_time) / num_runs
        
        print(f'Configuration: {config}')
        print(f'Average execution time: {avg_time:.4f} seconds')
        print()


if __name__ == '__main__':
    main()
