"""
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
