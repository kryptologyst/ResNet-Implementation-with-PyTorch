"""
Advanced Logging and Experiment Tracking
========================================

Comprehensive logging system with support for:
- File and console logging
- TensorBoard integration
- Weights & Biases integration
- Experiment tracking
- Model checkpointing
- Performance monitoring
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentLogger:
    """Comprehensive experiment logging system"""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        log_level: str = "INFO"
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Setup TensorBoard
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
        else:
            self.tb_writer = None
        
        # Setup Weights & Biases
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "resnet-experiments",
                entity=wandb_entity,
                name=experiment_name,
                dir=str(self.experiment_dir)
            )
        
        # Experiment metadata
        self.start_time = time.time()
        self.metrics_history = {
            'train_loss': [], 'train_acc': [], 'train_top5_acc': [],
            'val_loss': [], 'val_acc': [], 'val_top5_acc': [],
            'learning_rate': [], 'epoch_time': []
        }
        
        self.logger.info(f"🚀 Experiment '{experiment_name}' started")
        self.logger.info(f"📁 Log directory: {self.experiment_dir}")
        self.logger.info(f"📊 TensorBoard: {'✅' if self.use_tensorboard else '❌'}")
        self.logger.info(f"🔮 Weights & Biases: {'✅' if self.use_wandb else '❌'}")
    
    def setup_logging(self, log_level: str):
        """Setup file and console logging"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(
            self.experiment_dir / "experiment.log",
            mode='w'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info("📋 Configuration logged")
        
        # Log to W&B
        if self.use_wandb:
            wandb.config.update(config)
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model information"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        self.logger.info(f"🧠 Model Info:")
        self.logger.info(f"   Total parameters: {total_params:,}")
        self.logger.info(f"   Trainable parameters: {trainable_params:,}")
        self.logger.info(f"   Model size: {model_info['model_size_mb']:.2f} MB")
        
        # Log to W&B
        if self.use_wandb:
            wandb.config.update(model_info)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], learning_rate: float):
        """Log epoch metrics"""
        epoch_time = time.time() - self.start_time
        
        # Update history
        self.metrics_history['train_loss'].append(metrics.get('train_loss', 0))
        self.metrics_history['train_acc'].append(metrics.get('train_acc', 0))
        self.metrics_history['train_top5_acc'].append(metrics.get('train_top5_acc', 0))
        self.metrics_history['val_loss'].append(metrics.get('val_loss', 0))
        self.metrics_history['val_acc'].append(metrics.get('val_acc', 0))
        self.metrics_history['val_top5_acc'].append(metrics.get('val_top5_acc', 0))
        self.metrics_history['learning_rate'].append(learning_rate)
        self.metrics_history['epoch_time'].append(epoch_time)
        
        # Log to console
        self.logger.info(
            f"📊 Epoch {epoch:3d} - "
            f"Train Loss: {metrics.get('train_loss', 0):.4f}, "
            f"Train Acc: {metrics.get('train_acc', 0):.4f} - "
            f"Val Loss: {metrics.get('val_loss', 0):.4f}, "
            f"Val Acc: {metrics.get('val_acc', 0):.4f} - "
            f"LR: {learning_rate:.6f}"
        )
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('Loss/Train', metrics.get('train_loss', 0), epoch)
            self.tb_writer.add_scalar('Loss/Validation', metrics.get('val_loss', 0), epoch)
            self.tb_writer.add_scalar('Accuracy/Train', metrics.get('train_acc', 0), epoch)
            self.tb_writer.add_scalar('Accuracy/Validation', metrics.get('val_acc', 0), epoch)
            self.tb_writer.add_scalar('Learning_Rate', learning_rate, epoch)
            
            if metrics.get('train_top5_acc', 0) > 0:
                self.tb_writer.add_scalar('Top5_Accuracy/Train', metrics.get('train_top5_acc', 0), epoch)
            if metrics.get('val_top5_acc', 0) > 0:
                self.tb_writer.add_scalar('Top5_Accuracy/Validation', metrics.get('val_top5_acc', 0), epoch)
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': metrics.get('train_loss', 0),
                'val_loss': metrics.get('val_loss', 0),
                'train_acc': metrics.get('train_acc', 0),
                'val_acc': metrics.get('val_acc', 0),
                'learning_rate': learning_rate
            })
    
    def log_test_results(self, test_metrics: Dict[str, float]):
        """Log final test results"""
        self.logger.info("🎯 Final Test Results:")
        self.logger.info(f"   Test Loss: {test_metrics.get('loss', 0):.4f}")
        self.logger.info(f"   Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        if test_metrics.get('top5_accuracy', 0) > 0:
            self.logger.info(f"   Test Top-5 Accuracy: {test_metrics.get('top5_accuracy', 0):.4f}")
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('Test/Loss', test_metrics.get('loss', 0))
            self.tb_writer.add_scalar('Test/Accuracy', test_metrics.get('accuracy', 0))
            if test_metrics.get('top5_accuracy', 0) > 0:
                self.tb_writer.add_scalar('Test/Top5_Accuracy', test_metrics.get('top5_accuracy', 0))
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'test_loss': test_metrics.get('loss', 0),
                'test_accuracy': test_metrics.get('accuracy', 0),
                'test_top5_accuracy': test_metrics.get('top5_accuracy', 0)
            })
    
    def log_model_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                           epoch: int, metrics: Dict[str, float]):
        """Log model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def log_best_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                      epoch: int, metrics: Dict[str, float]):
        """Log best model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'is_best': True
        }
        
        checkpoint_path = self.experiment_dir / "best_model.pth"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"🏆 Best model saved: {checkpoint_path}")
    
    def log_training_history(self):
        """Log complete training history"""
        history_path = self.experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"📈 Training history saved: {history_path}")
    
    def log_experiment_summary(self):
        """Log experiment summary"""
        total_time = time.time() - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_time_hours': total_time / 3600,
            'total_epochs': len(self.metrics_history['train_loss']),
            'best_train_acc': max(self.metrics_history['train_acc']) if self.metrics_history['train_acc'] else 0,
            'best_val_acc': max(self.metrics_history['val_acc']) if self.metrics_history['val_acc'] else 0,
            'final_train_loss': self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else 0,
            'final_val_loss': self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("📋 Experiment Summary:")
        self.logger.info(f"   Total time: {summary['total_time_hours']:.2f} hours")
        self.logger.info(f"   Total epochs: {summary['total_epochs']}")
        self.logger.info(f"   Best train accuracy: {summary['best_train_acc']:.4f}")
        self.logger.info(f"   Best validation accuracy: {summary['best_val_acc']:.4f}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log(summary)
    
    def close(self):
        """Close all logging resources"""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        self.logger.info("✅ Experiment logging completed")


class PerformanceMonitor:
    """Monitor system performance during training"""
    
    def __init__(self):
        self.start_time = time.time()
        self.epoch_times = []
        self.memory_usage = []
    
    def start_epoch(self):
        """Mark start of epoch"""
        self.epoch_start = time.time()
    
    def end_epoch(self):
        """Mark end of epoch and record metrics"""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Get memory usage if available
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            self.memory_usage.append(memory_percent)
        except ImportError:
            pass
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        stats = {
            'total_time': time.time() - self.start_time,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'total_epochs': len(self.epoch_times)
        }
        
        if self.memory_usage:
            stats['avg_memory_usage'] = np.mean(self.memory_usage)
            stats['max_memory_usage'] = np.max(self.memory_usage)
        
        return stats


def setup_experiment_logging(
    experiment_name: str,
    config: Dict[str, Any],
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None
) -> ExperimentLogger:
    """Setup experiment logging with configuration"""
    
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project
    )
    
    logger.log_config(config)
    
    return logger


def main():
    """Example usage of logging system"""
    # Example configuration
    config = {
        'model': {'name': 'resnet18', 'num_classes': 10},
        'training': {'epochs': 200, 'learning_rate': 0.1},
        'data': {'batch_size': 128, 'dataset': 'cifar10'}
    }
    
    # Setup logging
    logger = setup_experiment_logging(
        experiment_name="test_experiment",
        config=config,
        use_tensorboard=True,
        use_wandb=False
    )
    
    # Simulate training
    for epoch in range(5):
        metrics = {
            'train_loss': 1.0 - epoch * 0.1,
            'train_acc': epoch * 0.1,
            'val_loss': 1.1 - epoch * 0.1,
            'val_acc': epoch * 0.09
        }
        logger.log_epoch(epoch, metrics, learning_rate=0.1)
    
    # Log test results
    test_metrics = {'loss': 0.5, 'accuracy': 0.85}
    logger.log_test_results(test_metrics)
    
    # Close logging
    logger.close()
    
    print("✅ Logging example completed")


if __name__ == "__main__":
    main()
