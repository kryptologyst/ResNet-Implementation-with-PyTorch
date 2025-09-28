"""
Model Evaluation and Visualization Tools
========================================

Comprehensive evaluation and visualization tools for ResNet models including:
- Confusion matrix visualization
- Training curves plotting
- Model performance analysis
- Feature visualization
- Grad-CAM implementation
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2

from resnet import create_model


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: Optional[List[str]] = None):
        self.model = model
        self.device = device
        self.class_names = class_names or [f"Class {i}" for i in range(10)]
        self.model.eval()
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_losses = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                # Get predictions and probabilities
                probabilities = F.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_losses.append(loss.item())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        avg_loss = np.mean(all_losses)
        
        # Top-5 accuracy if applicable
        if len(self.class_names) > 5:
            top5_correct = 0
            for i, (pred_probs, target) in enumerate(zip(all_probabilities, all_targets)):
                top5_preds = np.argsort(pred_probs)[-5:]
                if target in top5_preds:
                    top5_correct += 1
            top5_accuracy = top5_correct / len(all_targets)
        else:
            top5_accuracy = accuracy
        
        results = {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'loss': avg_loss,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        return results
    
    def plot_confusion_matrix(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(results['targets'], results['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot classification report"""
        report = classification_report(
            results['targets'], 
            results['predictions'], 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Extract metrics for plotting
        classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1_score = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Classification Report')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot prediction confidence distribution"""
        probabilities = np.array(results['probabilities'])
        max_probs = np.max(probabilities, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassifications(self, results: Dict[str, Any], test_loader: DataLoader, 
                                 num_samples: int = 10) -> List[Dict[str, Any]]:
        """Analyze misclassified samples"""
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        probabilities = np.array(results['probabilities'])
        
        # Find misclassified samples
        misclassified_indices = np.where(predictions != targets)[0]
        
        misclassified_samples = []
        for i, idx in enumerate(misclassified_indices[:num_samples]):
            sample = {
                'index': idx,
                'true_label': self.class_names[targets[idx]],
                'predicted_label': self.class_names[predictions[idx]],
                'confidence': probabilities[idx][predictions[idx]],
                'true_confidence': probabilities[idx][targets[idx]]
            }
            misclassified_samples.append(sample)
        
        return misclassified_samples


class TrainingVisualizer:
    """Visualize training progress and results"""
    
    def __init__(self, history: Dict[str, List[float]]):
        self.history = history
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Training Accuracy', alpha=0.8)
        axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy', alpha=0.8)
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 accuracy curves (if available)
        if 'train_top5_acc' in self.history and 'val_top5_acc' in self.history:
            axes[1, 0].plot(self.history['train_top5_acc'], label='Training Top-5 Acc', alpha=0.8)
            axes[1, 0].plot(self.history['val_top5_acc'], label='Validation Top-5 Acc', alpha=0.8)
            axes[1, 0].set_title('Top-5 Accuracy Curves')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-5 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Top-5 Accuracy\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Top-5 Accuracy Curves')
        
        # Learning rate curve
        axes[1, 1].plot(self.history['learning_rate'], label='Learning Rate', alpha=0.8)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_loss_comparison(self, save_path: Optional[str] = None):
        """Plot training vs validation loss comparison"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss', alpha=0.8)
        plt.plot(self.history['val_loss'], label='Validation Loss', alpha=0.8)
        plt.fill_between(range(len(self.history['train_loss'])), 
                        self.history['train_loss'], alpha=0.3)
        plt.fill_between(range(len(self.history['val_loss'])), 
                        self.history['val_loss'], alpha=0.3)
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_accuracy_comparison(self, save_path: Optional[str] = None):
        """Plot training vs validation accuracy comparison"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_acc'], label='Training Accuracy', alpha=0.8)
        plt.plot(self.history['val_acc'], label='Validation Accuracy', alpha=0.8)
        plt.fill_between(range(len(self.history['train_acc'])), 
                        self.history['train_acc'], alpha=0.3)
        plt.fill_between(range(len(self.history['val_acc'])), 
                        self.history['val_acc'], alpha=0.3)
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) implementation"""
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM for the given input"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, input_tensor: torch.Tensor, original_image: np.ndarray, 
                     class_idx: Optional[int] = None, save_path: Optional[str] = None):
        """Visualize Grad-CAM overlay on original image"""
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def __del__(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()


def load_model_and_history(checkpoint_path: str, model_name: str = 'resnet18', 
                          num_classes: int = 10) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Load trained model and training history"""
    # Create model
    model = create_model(model_name, num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load history
    history = checkpoint.get('history', {})
    
    return model, history


def main():
    """Example usage of evaluation and visualization tools"""
    # Load model and history (assuming you have a trained model)
    checkpoint_path = './checkpoints/best_model.pth'
    
    if os.path.exists(checkpoint_path):
        model, history = load_model_and_history(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create evaluator
        evaluator = ModelEvaluator(model, device, class_names)
        
        # Load test data (you would need to create this)
        # test_loader = get_test_loader()
        
        # Evaluate model
        # results = evaluator.evaluate_model(test_loader)
        
        # Visualize training curves
        visualizer = TrainingVisualizer(history)
        visualizer.plot_training_curves()
        
        print("✅ Evaluation and visualization tools ready!")
    else:
        print("❌ No trained model found. Please train a model first.")


if __name__ == "__main__":
    main()
