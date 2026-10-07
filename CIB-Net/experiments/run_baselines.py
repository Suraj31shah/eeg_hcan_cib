import sys
import os
import yaml
import argparse
import torch

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.deap_loader import DEAPDataset
from data.preprocessing import preprocess_eeg
from training.loso_evaluator import run_loso_evaluation, run_leaky_evaluation
from models.baselines.plain_cnn import PlainCNN
from models.baselines.eegnet import EEGNet
from models.baselines.acrnn import ACRNN

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'eegnet', 'acrnn'], help='Baseline model to run')
    parser.add_argument('--eval', type=str, required=True, choices=['loso', 'leaky', 'both'], help='Evaluation protocol')
    args = parser.parse_args()
    
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading DEAP Dataset...")
    dataset = DEAPDataset(
        data_path=config['data_path'],
        target=config['target'],
        preprocess_fn=lambda x: preprocess_eeg(
            x, 
            fs=config['sample_rate'], 
            baseline_sec=config['baseline_sec']
        )
    )
    
    if len(dataset) == 0:
        print("Dataset is empty. Please check your data_path in config.yaml.")
        return
        
    model_map = {
        'cnn': PlainCNN,
        'eegnet': EEGNet,
        'acrnn': ACRNN
    }
    
    model_class = model_map[args.model]
    model_kwargs = {
        'n_channels': config['n_channels'],
        'n_classes': config['n_classes']
    }
    
    print(f"\nEvaluating Baseline: {args.model.upper()}")
    
    if args.eval in ['leaky', 'both']:
        run_leaky_evaluation(
            dataset=dataset,
            model_class=model_class,
            model_kwargs=model_kwargs,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            lr=config['training']['lr'],
            device=device
        )
        
    if args.eval in ['loso', 'both']:
        run_loso_evaluation(
            dataset=dataset,
            model_class=model_class,
            model_kwargs=model_kwargs,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            lr=config['training']['lr'],
            device=device
        )

if __name__ == '__main__':
    main()
