import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.dreamer_loader import WindowedDREAMERDataset
from models.hcan_cib_net import HCANCIBNet
from training.loso_evaluator import run_hcan_cib_loso

def main():
    print("Initializing HCAN-CIB Unified Pipeline for DREAMER...")
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'hcan_cib_config.yaml')
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    dataset = WindowedDREAMERDataset(
        data_path=config['data_path'],
        target=config['target'],
        binarize=True,
        window_sec=config['window_sec'],
        overlap_sec=config['overlap_sec']
    )
    
    print(f"Loaded DREAMER Dataset. {len(dataset)} windows prepared.")
    
    # Run the evaluation
    run_hcan_cib_loso(dataset, HCANCIBNet, config_path)

if __name__ == "__main__":
    main()
