import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.windowed_loader import WindowedDEAPDataset
from models.hcan_cib_net import HCANCIBNet
from training.loso_evaluator import run_hcan_cib_loso

def main():
    print("Initializing HCAN-CIB Unified Pipeline...")
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'hcan_cib_config.yaml')
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    dataset = WindowedDEAPDataset(
        data_path=config['data_path'],
        target=config['target'],
        binarize=config['binarize'],
        window_sec=config['window_sec'],
        overlap_sec=config['overlap_sec'],
        fs=config['sample_rate'],
        baseline_sec=config['baseline_sec']
    )
    
    print(f"Loaded DEAP Dataset. {len(dataset)} windows prepared.")
    
    # Run the evaluation
    run_hcan_cib_loso(dataset, HCANCIBNet, config_path)

if __name__ == "__main__":
    main()
