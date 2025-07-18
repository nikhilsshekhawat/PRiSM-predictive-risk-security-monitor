"""
Security Scanning using AI - Main Training and Inference Script
==========================================================

This script provides the main interface for training and running inference
with the security scanning AI system for detecting explosives, ammunition,
and other harmful objects.

Author: Nikhil Shekhawat
"""

import torch
import torch.nn as nn
import argparse
import yaml
import json
import logging
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project modules to path
sys.path.append(str(Path(__file__).parent))

from models.yolov8_security import YOLOv8Security
from models.faster_rcnn_security import FasterRCNNSecurity
from models.ensemble_model import EnsembleModel
from training.trainer import SecurityTrainer
from training.hyperparameter_tuning import HyperparameterTuner
from data.datasets import SecurityDataset
from utils.logging_utils import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Security Scanning AI - Training and Inference'
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'infer', 'tune', 'evaluate'],
                       help='Mode: train, infer, tune, or evaluate')

    # Configuration
    parser.add_argument('--config', type=str, default='security_scanning_config.json',
                       help='Path to configuration file')

    # Data
    parser.add_argument('--data', type=str, default='./data/',
                       help='Path to dataset directory')

    # Model
    parser.add_argument('--model', type=str, default='yolov8',
                       choices=['yolov8', 'faster_rcnn', 'ensemble'],
                       help='Model architecture to use')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    # Inference parameters
    parser.add_argument('--source', type=str, default='',
                       help='Source for inference (image/video path)')
    parser.add_argument('--weights', type=str, default='',
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    # Output
    parser.add_argument('--output', type=str, default='./outputs/',
                       help='Output directory')

    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in configuration file {config_path}")
        sys.exit(1)

def main():
    """Main function to execute training or inference."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.epochs != 200:
        config['training']['epochs'] = args.epochs
    if args.batch_size != 16:
        config['data']['batch_size'] = args.batch_size
    if args.lr != 0.001:
        config['training']['learning_rate'] = args.lr

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        logger.info("Starting training mode...")

        # Initialize trainer
        trainer = SecurityTrainer(config, device)

        # Load dataset
        dataset = SecurityDataset(
            data_path=args.data,
            config=config,
            mode='train'
        )

        # Select model
        if args.model == 'yolov8':
            model = YOLOv8Security(config)
        elif args.model == 'faster_rcnn':
            model = FasterRCNNSecurity(config)
        else:
            model = EnsembleModel(config)

        # Train model
        trainer.train(model, dataset, args.output)

    elif args.mode == 'infer':
        logger.info("Starting inference mode...")

        if not args.source:
            logger.error("Please provide --source for inference")
            sys.exit(1)

        if not args.weights:
            logger.error("Please provide --weights for inference")
            sys.exit(1)

        # Load model
        if args.model == 'yolov8':
            model = YOLOv8Security.load_from_checkpoint(args.weights)
        elif args.model == 'faster_rcnn':
            model = FasterRCNNSecurity.load_from_checkpoint(args.weights)
        else:
            model = EnsembleModel.load_from_checkpoint(args.weights)

        model.to(device)
        model.eval()

        # Run inference
        from inference.detector import SecurityDetector
        detector = SecurityDetector(model, config, device)
        detector.detect(args.source, args.output, args.conf, args.iou)

    elif args.mode == 'tune':
        logger.info("Starting hyperparameter tuning mode...")

        # Initialize tuner
        tuner = HyperparameterTuner(config, device)

        # Load dataset
        dataset = SecurityDataset(
            data_path=args.data,
            config=config,
            mode='train'
        )

        # Run tuning
        best_params = tuner.tune(dataset, args.output)
        logger.info(f"Best parameters: {best_params}")

    elif args.mode == 'evaluate':
        logger.info("Starting evaluation mode...")

        if not args.weights:
            logger.error("Please provide --weights for evaluation")
            sys.exit(1)

        # Load model
        if args.model == 'yolov8':
            model = YOLOv8Security.load_from_checkpoint(args.weights)
        elif args.model == 'faster_rcnn':
            model = FasterRCNNSecurity.load_from_checkpoint(args.weights)
        else:
            model = EnsembleModel.load_from_checkpoint(args.weights)

        model.to(device)
        model.eval()

        # Load test dataset
        test_dataset = SecurityDataset(
            data_path=args.data,
            config=config,
            mode='test'
        )

        # Run evaluation
        from training.evaluation import SecurityEvaluator
        evaluator = SecurityEvaluator(config, device)
        metrics = evaluator.evaluate(model, test_dataset)

        logger.info(f"Evaluation results: {metrics}")

        # Save results
        with open(f"{args.output}/evaluation_results.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()
