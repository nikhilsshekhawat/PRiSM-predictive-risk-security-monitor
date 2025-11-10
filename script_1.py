# Create the main training script and key implementation files
import os

# Create main.py - Main training and inference script
main_py_content = """
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
'''

# Create YOLOv8 security model implementation
yolov8_security_content = '''"""
YOLOv8 Security Model Implementation
===================================

Custom YOLOv8 implementation optimized for security scanning applications.
Includes modifications for detecting explosives, ammunition, and harmful objects.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class YOLOv8Security(nn.Module):
    """
    YOLOv8 model specialized for security scanning applications.
    
    Features:
    - Multi-scale detection for small harmful objects
    - Custom anchor boxes optimized for security items
    - Enhanced feature pyramid network for better detection
    - Transfer learning from COCO pre-trained weights
    """
    
    def __init__(self, config: Dict):
        """
        Initialize YOLOv8 Security model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(YOLOv8Security, self).__init__()
        
        self.config = config
        self.num_classes = config['models']['num_classes']
        self.model_size = config['models']['architectures'][0]  # Default to first architecture
        
        # Initialize base YOLOv8 model
        self.model = YOLO(f'{self.model_size}.pt')
        
        # Class names for security scanning
        self.class_names = [
            'gun', 'knife', 'explosive', 'ammunition', 'grenade',
            'scissors', 'razor', 'lighter', 'battery', 'liquid',
            'electronics', 'suspicious_object'
        ]
        
        # Custom anchor boxes for security items
        self.anchor_boxes = self._get_security_anchors()
        
        # Data augmentation for training
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation((-10, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"YOLOv8Security model initialized with {self.num_classes} classes")
    
    def _get_security_anchors(self) -> List[Tuple[int, int]]:
        """
        Generate custom anchor boxes optimized for security items.
        
        Returns:
            List of (width, height) tuples for anchor boxes
        """
        # Anchor boxes designed for typical security items
        anchors = [
            (10, 20),   # Small knife/blade
            (15, 30),   # Larger knife
            (20, 40),   # Gun barrel
            (30, 15),   # Handgun
            (40, 20),   # Rifle
            (25, 25),   # Explosive device
            (15, 15),   # Ammunition
            (35, 35),   # Large explosive
            (12, 25),   # Scissors
            (8, 15),    # Razor blade
            (18, 12),   # Lighter
            (22, 18),   # Battery
        ]
        
        return anchors
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Model predictions
        """
        return self.model(x)
    
    def train_step(self, batch: Dict) -> Dict:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch containing images and labels
            
        Returns:
            Dictionary containing loss values
        """
        images = batch['images']
        labels = batch['labels']
        
        # Forward pass
        results = self.model.train(
            data=batch,
            epochs=1,
            imgsz=self.config['data']['image_size'][0],
            batch=self.config['data']['batch_size'],
            verbose=False
        )
        
        return {
            'total_loss': results.results_dict.get('train/box_loss', 0) + 
                         results.results_dict.get('train/cls_loss', 0) + 
                         results.results_dict.get('train/dfl_loss', 0),
            'box_loss': results.results_dict.get('train/box_loss', 0),
            'cls_loss': results.results_dict.get('train/cls_loss', 0),
            'dfl_loss': results.results_dict.get('train/dfl_loss', 0)
        }
    
    def validate_step(self, batch: Dict) -> Dict:
        """
        Perform a single validation step.
        
        Args:
            batch: Validation batch containing images and labels
            
        Returns:
            Dictionary containing validation metrics
        """
        images = batch['images']
        labels = batch['labels']
        
        # Forward pass
        results = self.model.val(
            data=batch,
            imgsz=self.config['data']['image_size'][0],
            batch=self.config['data']['batch_size'],
            verbose=False
        )
        
        return {
            'val_loss': results.results_dict.get('val/box_loss', 0) + 
                       results.results_dict.get('val/cls_loss', 0) + 
                       results.results_dict.get('val/dfl_loss', 0),
            'mAP50': results.results_dict.get('val/mAP50', 0),
            'mAP50-95': results.results_dict.get('val/mAP50-95', 0),
            'precision': results.results_dict.get('val/precision', 0),
            'recall': results.results_dict.get('val/recall', 0)
        }
    
    def predict(self, images: torch.Tensor, conf_threshold: float = 0.25, 
                iou_threshold: float = 0.45) -> List[Dict]:
        """
        Make predictions on input images.
        
        Args:
            images: Input tensor of shape (batch_size, channels, height, width)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of prediction dictionaries
        """
        results = self.model.predict(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        predictions = []
        for result in results:
            pred = {
                'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([]),
                'scores': result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([]),
                'labels': result.boxes.cls.cpu().numpy() if result.boxes is not None else np.array([]),
                'class_names': [self.class_names[int(cls)] for cls in result.boxes.cls.cpu().numpy()] if result.boxes is not None else []
            }
            predictions.append(pred)
        
        return predictions
    
    def detect_security_threats(self, image_path: str, 
                              conf_threshold: float = 0.25) -> Dict:
        """
        Detect security threats in an image.
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing detection results and threat assessment
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        results = self.model.predict(
            image_rgb,
            conf=conf_threshold,
            verbose=False
        )
        
        # Process results
        detections = []
        threat_level = "LOW"
        
        for result in results:
            if result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    class_name = self.class_names[int(cls)]
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': box.cpu().numpy().tolist(),
                        'threat_level': self._assess_threat_level(class_name, float(conf))
                    }
                    detections.append(detection)
                    
                    # Update overall threat level
                    if detection['threat_level'] == "HIGH":
                        threat_level = "HIGH"
                    elif detection['threat_level'] == "MEDIUM" and threat_level == "LOW":
                        threat_level = "MEDIUM"
        
        return {
            'image_path': image_path,
            'detections': detections,
            'threat_level': threat_level,
            'num_detections': len(detections)
        }
    
    def _assess_threat_level(self, class_name: str, confidence: float) -> str:
        """
        Assess threat level based on detected object class and confidence.
        
        Args:
            class_name: Name of detected object class
            confidence: Detection confidence score
            
        Returns:
            Threat level: "LOW", "MEDIUM", or "HIGH"
        """
        # High-threat objects
        high_threat = ['gun', 'explosive', 'ammunition', 'grenade']
        
        # Medium-threat objects
        medium_threat = ['knife', 'scissors', 'razor']
        
        # Low-threat objects
        low_threat = ['lighter', 'battery', 'electronics']
        
        if class_name in high_threat:
            return "HIGH" if confidence > 0.5 else "MEDIUM"
        elif class_name in medium_threat:
            return "MEDIUM" if confidence > 0.6 else "LOW"
        else:
            return "LOW"
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict = None):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dictionary
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'class_names': self.class_names
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_from_checkpoint(cls, path: str):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Initialize model with saved config
        model = cls(checkpoint['config'])
        
        # Load model state
        model.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load class names
        model.class_names = checkpoint['class_names']
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def get_model_info(self) -> Dict:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': 'YOLOv8Security',
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

def create_yolov8_security_model(config: Dict) -> YOLOv8Security:
    """
    Factory function to create YOLOv8 security model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        YOLOv8Security model instance
    """
    return YOLOv8Security(config)

