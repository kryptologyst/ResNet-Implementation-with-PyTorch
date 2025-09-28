"""
Training Pipeline for ResNet Implementation
==========================================

A comprehensive training pipeline with modern techniques including:
- Data augmentation and preprocessing
- Learning rate scheduling
- Mixed precision training
- Model checkpointing
- Early stopping
- Comprehensive logging
"""

import os
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from resnet import create_model, MODEL_REGISTRY


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save the best model weights"""
        self.best_weights = model.state_dict().copy()


class MetricsTracker:
    """Track training and validation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.accuracies = []
        self.top5_accuracies = []
    
    def update(self, loss: float, accuracy: float, top5_accuracy: float = 0.0):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.top5_accuracies.append(top5_accuracy)
    
    def get_averages(self) -> Dict[str, float]:
        return {
            'loss': np.mean(self.losses),
            'accuracy': np.mean(self.accuracies),
            'top5_accuracy': np.mean(self.top5_accuracies)
        }


class ResNetTrainer:
    """Main training class for ResNet models"""
    
    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 10,
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        **model_kwargs
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Create model
        self.model = create_model(model_name, num_classes=num_classes, **model_kwargs)
        self.model = self.model.to(self.device)
        
        # Initialize metrics
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_top5_acc': [],
            'val_loss': [], 'val_acc': [], 'val_top5_acc': [],
            'learning_rate': []
        }
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_data_loaders(
        self,
        dataset_name: str = 'cifar10',
        batch_size: int = 128,
        num_workers: int = 4,
        data_dir: str = './data'
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders for training, validation, and testing"""
        
        # Define transforms
        if dataset_name.lower() == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            num_classes = 10
        elif dataset_name.lower() == 'cifar100':
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
            num_classes = 100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1)
        ])
        
        # Validation/test transforms
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Load datasets
        if dataset_name.lower() == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=train_transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=val_transform
            )
        elif dataset_name.lower() == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=train_transform
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=val_transform
            )
        
        # Split training data into train/validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Apply transforms to validation set
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Calculate metrics
            pred = output.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(target.view_as(pred)).sum().item() / len(data)
            
            # Top-5 accuracy for datasets with more than 5 classes
            if output.size(1) > 5:
                _, top5_pred = output.topk(5, dim=1)
                top5_accuracy = top5_pred.eq(target.view(-1, 1)).sum().item() / len(data)
            else:
                top5_accuracy = accuracy
            
            self.train_metrics.update(loss.item(), accuracy, top5_accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
        
        return self.train_metrics.get_averages()
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.mixed_precision:
                    with autocast():
                        output = self.model(data)
                        loss = criterion(output, target)
                else:
                    output = self.model(data)
                    loss = criterion(output, target)
                
                # Calculate metrics
                pred = output.argmax(dim=1, keepdim=True)
                accuracy = pred.eq(target.view_as(pred)).sum().item() / len(data)
                
                # Top-5 accuracy
                if output.size(1) > 5:
                    _, top5_pred = output.topk(5, dim=1)
                    top5_accuracy = top5_pred.eq(target.view(-1, 1)).sum().item() / len(data)
                else:
                    top5_accuracy = accuracy
                
                self.val_metrics.update(loss.item(), accuracy, top5_accuracy)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
        
        return self.val_metrics.get_averages()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        scheduler_type: str = 'cosine',
        save_dir: str = './checkpoints',
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """Main training loop"""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
        elif scheduler_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)
        else:
            scheduler = None
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Training loop
        best_val_acc = 0.0
        start_time = time.time()
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.mixed_precision}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_top5_acc'].append(train_metrics['top5_accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_top5_acc'].append(val_metrics['top5_accuracy'])
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f} - "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_acc': val_metrics['accuracy'],
                    'train_acc': train_metrics['accuracy'],
                    'history': self.history
                }, os.path.join(save_dir, 'best_model.pth'))
            
            # Early stopping
            if early_stopping(val_metrics['loss'], self.model):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save final model and history
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_acc': best_val_acc
        }, os.path.join(save_dir, 'final_model.pth'))
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        test_metrics = MetricsTracker()
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.mixed_precision:
                    with autocast():
                        output = self.model(data)
                        loss = nn.CrossEntropyLoss()(output, target)
                else:
                    output = self.model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                
                pred = output.argmax(dim=1, keepdim=True)
                accuracy = pred.eq(target.view_as(pred)).sum().item() / len(data)
                
                if output.size(1) > 5:
                    _, top5_pred = output.topk(5, dim=1)
                    top5_accuracy = top5_pred.eq(target.view(-1, 1)).sum().item() / len(data)
                else:
                    top5_accuracy = accuracy
                
                test_metrics.update(loss.item(), accuracy, top5_accuracy)
        
        results = test_metrics.get_averages()
        self.logger.info(f"Test Results - Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.4f}")
        
        return results


def main():
    """Main training script"""
    # Configuration
    config = {
        'model_name': 'resnet18',
        'dataset': 'cifar10',
        'num_classes': 10,
        'batch_size': 128,
        'epochs': 200,
        'learning_rate': 0.1,
        'weight_decay': 1e-4,
        'scheduler_type': 'cosine',
        'mixed_precision': True,
        'early_stopping_patience': 20
    }
    
    # Create trainer
    trainer = ResNetTrainer(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        mixed_precision=config['mixed_precision']
    )
    
    # Get data loaders
    train_loader, val_loader, test_loader = trainer.get_data_loaders(
        dataset_name=config['dataset'],
        batch_size=config['batch_size']
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler_type=config['scheduler_type'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Final test accuracy: {test_results['accuracy']:.4f}")
    print(f"📈 Best validation accuracy: {max(history['val_acc']):.4f}")


if __name__ == "__main__":
    main()
