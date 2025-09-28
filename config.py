"""
Configuration Management System
==============================

A comprehensive configuration management system for ResNet training and evaluation
with support for YAML configuration files, hyperparameter tuning, and experiment tracking.
"""

import os
import yaml
import json
import argparse
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    name: str = 'resnet18'
    num_classes: int = 10
    pretrained: bool = False
    zero_init_residual: bool = False
    groups: int = 1
    width_per_group: int = 64
    replace_stride_with_dilation: Optional[List[bool]] = None


@dataclass
class DataConfig:
    """Data configuration parameters"""
    dataset_name: str = 'cifar10'
    data_dir: str = './data'
    batch_size: int = 128
    num_workers: int = 4
    train_split: float = 0.8
    augmentation: bool = True
    normalize: bool = True


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    epochs: int = 200
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler_type: str = 'cosine'
    scheduler_params: Optional[Dict[str, Any]] = None
    mixed_precision: bool = True
    gradient_clipping: Optional[float] = None
    early_stopping_patience: int = 20
    save_every_n_epochs: int = 10


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration"""
    log_level: str = 'INFO'
    log_file: str = 'training.log'
    experiment_name: Optional[str] = None
    save_dir: str = './checkpoints'
    tensorboard_log_dir: str = './logs/tensorboard'
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    logging: LoggingConfig
    
    def __post_init__(self):
        """Post-initialization validation"""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        # Model validation
        valid_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if self.model.name not in valid_models:
            raise ValueError(f"Invalid model name: {self.model.name}. Valid options: {valid_models}")
        
        # Dataset validation
        valid_datasets = ['cifar10', 'cifar100']
        if self.data.dataset_name not in valid_datasets:
            raise ValueError(f"Invalid dataset: {self.data.dataset_name}. Valid options: {valid_datasets}")
        
        # Scheduler validation
        valid_schedulers = ['cosine', 'step', 'multistep', 'exponential', 'plateau']
        if self.training.scheduler_type not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {self.training.scheduler_type}. Valid options: {valid_schedulers}")
        
        # Log level validation
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.logging.log_level}. Valid options: {valid_log_levels}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for section, values in updates.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        logging.warning(f"Unknown parameter: {section}.{key}")


class ConfigManager:
    """Configuration manager with advanced features"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, path: str):
        """Load configuration from file"""
        self.config = Config.load(path)
        self.config_path = path
        return self.config
    
    def create_default_config(self) -> Config:
        """Create default configuration"""
        self.config = Config(
            model=ModelConfig(),
            data=DataConfig(),
            training=TrainingConfig(),
            logging=LoggingConfig()
        )
        return self.config
    
    def create_experiment_config(self, experiment_name: str, **overrides) -> Config:
        """Create configuration for a specific experiment"""
        config = self.create_default_config()
        
        # Set experiment name
        config.logging.experiment_name = experiment_name
        config.logging.save_dir = f'./experiments/{experiment_name}'
        
        # Apply overrides
        if overrides:
            config.update(overrides)
        
        return config
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration"""
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        save_path = path or self.config_path
        if save_path is None:
            raise ValueError("No save path specified")
        
        self.config.save(save_path)
    
    def get_hyperparameter_search_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space for optimization"""
        return {
            'model.name': ['resnet18', 'resnet34', 'resnet50'],
            'training.learning_rate': [0.01, 0.05, 0.1, 0.2],
            'training.weight_decay': [1e-5, 1e-4, 1e-3],
            'training.batch_size': [64, 128, 256],
            'training.scheduler_type': ['cosine', 'step', 'multistep'],
            'data.augmentation': [True, False]
        }
    
    def generate_config_variants(self, base_config: Config, 
                               search_space: Dict[str, List[Any]]) -> List[Config]:
        """Generate configuration variants for hyperparameter search"""
        variants = []
        
        # This is a simplified version - in practice, you'd use itertools.product
        # or a proper hyperparameter optimization library
        for param, values in search_space.items():
            for value in values:
                variant = Config(
                    model=ModelConfig(**base_config.model.__dict__),
                    data=DataConfig(**base_config.data.__dict__),
                    training=TrainingConfig(**base_config.training.__dict__),
                    logging=LoggingConfig(**base_config.logging.__dict__)
                )
                
                # Update the specific parameter
                section, param_name = param.split('.')
                section_obj = getattr(variant, section)
                setattr(section_obj, param_name, value)
                
                variants.append(variant)
        
        return variants


def create_config_from_args() -> Config:
    """Create configuration from command line arguments"""
    parser = argparse.ArgumentParser(description='ResNet Training Configuration')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet model architecture')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of output classes')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to store datasets')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'multistep', 'exponential', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training')
    
    # Logging arguments
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name of the experiment')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load from config file if provided
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config(
            model=ModelConfig(
                name=args.model,
                num_classes=args.num_classes
            ),
            data=DataConfig(
                dataset_name=args.dataset,
                batch_size=args.batch_size,
                data_dir=args.data_dir
            ),
            training=TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                scheduler_type=args.scheduler,
                mixed_precision=args.mixed_precision
            ),
            logging=LoggingConfig(
                experiment_name=args.experiment_name,
                save_dir=args.save_dir,
                log_level=args.log_level
            )
        )
    
    return config


def create_sample_configs():
    """Create sample configuration files"""
    configs_dir = Path('./configs')
    configs_dir.mkdir(exist_ok=True)
    
    # Default configuration
    default_config = Config(
        model=ModelConfig(),
        data=DataConfig(),
        training=TrainingConfig(),
        logging=LoggingConfig()
    )
    default_config.save(str(configs_dir / 'default.yaml'))
    
    # CIFAR-10 configuration
    cifar10_config = Config(
        model=ModelConfig(name='resnet18', num_classes=10),
        data=DataConfig(dataset_name='cifar10', batch_size=128),
        training=TrainingConfig(epochs=200, learning_rate=0.1),
        logging=LoggingConfig(experiment_name='cifar10_resnet18')
    )
    cifar10_config.save(str(configs_dir / 'cifar10.yaml'))
    
    # CIFAR-100 configuration
    cifar100_config = Config(
        model=ModelConfig(name='resnet50', num_classes=100),
        data=DataConfig(dataset_name='cifar100', batch_size=64),
        training=TrainingConfig(epochs=300, learning_rate=0.05),
        logging=LoggingConfig(experiment_name='cifar100_resnet50')
    )
    cifar100_config.save(str(configs_dir / 'cifar100.yaml'))
    
    # Fast training configuration
    fast_config = Config(
        model=ModelConfig(name='resnet18', num_classes=10),
        data=DataConfig(batch_size=256),
        training=TrainingConfig(epochs=50, learning_rate=0.2),
        logging=LoggingConfig(experiment_name='fast_training')
    )
    fast_config.save(str(configs_dir / 'fast.yaml'))
    
    print(f"✅ Sample configurations created in {configs_dir}")


def main():
    """Example usage of configuration management"""
    # Create sample configurations
    create_sample_configs()
    
    # Load a configuration
    config_manager = ConfigManager()
    config = config_manager.load_config('./configs/cifar10.yaml')
    
    print("📋 Loaded Configuration:")
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Batch Size: {config.data.batch_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Learning Rate: {config.training.learning_rate}")
    
    # Create experiment configuration
    experiment_config = config_manager.create_experiment_config(
        'my_experiment',
        training={'learning_rate': 0.05, 'epochs': 100}
    )
    
    print(f"\n🧪 Experiment Configuration:")
    print(f"Experiment Name: {experiment_config.logging.experiment_name}")
    print(f"Save Directory: {experiment_config.logging.save_dir}")
    print(f"Learning Rate: {experiment_config.training.learning_rate}")
    
    # Save experiment configuration
    experiment_config.save('./configs/my_experiment.yaml')
    print("✅ Experiment configuration saved")


if __name__ == "__main__":
    main()
