"""
ResNet Implementation Demo Script
=================================

A comprehensive demo script that showcases all features of the ResNet implementation:
- Model creation and testing
- Training pipeline
- Evaluation and visualization
- Web interface
- Configuration management
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resnet import create_model, MODEL_REGISTRY
from train import ResNetTrainer
from evaluate import ModelEvaluator, TrainingVisualizer
from config import Config, ConfigManager
from logger import setup_experiment_logging


def demo_model_creation():
    """Demo model creation and basic functionality"""
    print("🧠 Demo: Model Creation and Testing")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test all model architectures
    for model_name in MODEL_REGISTRY.keys():
        print(f"\n📊 Testing {model_name}...")
        
        # Create model
        model = create_model(model_name, num_classes=10)
        model = model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32).to(device)
        output = model(dummy_input)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   📈 Parameters: {param_count:,}")
        print(f"   💾 Model size: {param_count * 4 / (1024 * 1024):.2f} MB")
    
    print("\n✅ Model creation demo completed!")


def demo_training():
    """Demo training pipeline"""
    print("\n🎯 Demo: Training Pipeline")
    print("=" * 50)
    
    # Create trainer
    trainer = ResNetTrainer(
        model_name='resnet18',
        num_classes=10,
        mixed_precision=False  # Disable for demo
    )
    
    print(f"📊 Model: {trainer.model.__class__.__name__}")
    print(f"🔧 Device: {trainer.device}")
    print(f"⚡ Mixed precision: {trainer.mixed_precision}")
    
    # Get data loaders (this will download CIFAR-10)
    print("\n📥 Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = trainer.get_data_loaders(
        dataset_name='cifar10',
        batch_size=32,  # Smaller batch for demo
        num_workers=2
    )
    
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"📊 Test samples: {len(test_loader.dataset)}")
    
    # Quick training demo (just a few epochs)
    print("\n🚀 Starting quick training demo (5 epochs)...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # Just a few epochs for demo
        learning_rate=0.1,
        save_dir='./demo_checkpoints'
    )
    
    print(f"📈 Final training accuracy: {history['train_acc'][-1]:.4f}")
    print(f"📈 Final validation accuracy: {history['val_acc'][-1]:.4f}")
    
    # Evaluate on test set
    print("\n🎯 Evaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    print(f"🎯 Test accuracy: {test_results['accuracy']:.4f}")
    
    print("\n✅ Training demo completed!")


def demo_evaluation():
    """Demo evaluation and visualization"""
    print("\n📊 Demo: Evaluation and Visualization")
    print("=" * 50)
    
    # Create a simple model for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model('resnet18', num_classes=10)
    model = model.to(device)
    
    # Create evaluator
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    evaluator = ModelEvaluator(model, device, class_names)
    
    print("📊 Model evaluator created")
    print(f"📊 Device: {evaluator.device}")
    print(f"📊 Classes: {len(evaluator.class_names)}")
    
    # Demo training visualization
    print("\n📈 Demo: Training Visualization")
    demo_history = {
        'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3],
        'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4],
        'train_acc': [0.3, 0.5, 0.7, 0.8, 0.9],
        'val_acc': [0.2, 0.4, 0.6, 0.75, 0.85],
        'learning_rate': [0.1, 0.08, 0.06, 0.04, 0.02]
    }
    
    visualizer = TrainingVisualizer(demo_history)
    print("📈 Training visualizer created")
    print(f"📈 Epochs tracked: {len(demo_history['train_loss'])}")
    
    print("\n✅ Evaluation demo completed!")


def demo_configuration():
    """Demo configuration management"""
    print("\n⚙️ Demo: Configuration Management")
    print("=" * 50)
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Create default configuration
    config = config_manager.create_default_config()
    print("📋 Default configuration created")
    print(f"📋 Model: {config.model.name}")
    print(f"📋 Dataset: {config.data.dataset_name}")
    print(f"📋 Epochs: {config.training.epochs}")
    
    # Create experiment configuration
    experiment_config = config_manager.create_experiment_config(
        'demo_experiment',
        training={'learning_rate': 0.05, 'epochs': 100}
    )
    print("\n🧪 Experiment configuration created")
    print(f"🧪 Experiment name: {experiment_config.logging.experiment_name}")
    print(f"🧪 Learning rate: {experiment_config.training.learning_rate}")
    print(f"🧪 Epochs: {experiment_config.training.epochs}")
    
    # Save configuration
    config_path = './demo_config.yaml'
    experiment_config.save(config_path)
    print(f"💾 Configuration saved to: {config_path}")
    
    # Load configuration
    loaded_config = Config.load(config_path)
    print(f"📥 Configuration loaded from: {config_path}")
    print(f"📥 Model: {loaded_config.model.name}")
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)
    
    print("\n✅ Configuration demo completed!")


def demo_logging():
    """Demo logging and experiment tracking"""
    print("\n📝 Demo: Logging and Experiment Tracking")
    print("=" * 50)
    
    # Setup experiment logging
    config = {
        'model': {'name': 'resnet18', 'num_classes': 10},
        'training': {'epochs': 5, 'learning_rate': 0.1},
        'data': {'batch_size': 32, 'dataset': 'cifar10'}
    }
    
    logger = setup_experiment_logging(
        experiment_name="demo_experiment",
        config=config,
        use_tensorboard=False,  # Disable for demo
        use_wandb=False  # Disable for demo
    )
    
    print("📝 Experiment logger created")
    print(f"📝 Experiment name: {logger.experiment_name}")
    print(f"📝 Log directory: {logger.experiment_dir}")
    
    # Simulate some logging
    logger.logger.info("Demo experiment started")
    
    # Simulate epoch logging
    for epoch in range(3):
        metrics = {
            'train_loss': 1.0 - epoch * 0.2,
            'train_acc': epoch * 0.2,
            'val_loss': 1.1 - epoch * 0.2,
            'val_acc': epoch * 0.18
        }
        logger.log_epoch(epoch, metrics, learning_rate=0.1)
    
    # Log test results
    test_metrics = {'loss': 0.5, 'accuracy': 0.85}
    logger.log_test_results(test_metrics)
    
    # Close logging
    logger.close()
    
    print("\n✅ Logging demo completed!")


def demo_web_interface():
    """Demo web interface setup"""
    print("\n🌐 Demo: Web Interface")
    print("=" * 50)
    
    print("🌐 Streamlit web interface is available!")
    print("🌐 To run the web interface:")
    print("   streamlit run app.py")
    print("🌐 Then open your browser to: http://localhost:8501")
    
    print("\n🌐 Web interface features:")
    print("   🔮 Real-time image classification")
    print("   📊 Model comparison tools")
    print("   📈 Training progress monitoring")
    print("   🔥 Grad-CAM visualization")
    print("   📊 Confidence score analysis")
    
    print("\n✅ Web interface demo completed!")


def run_full_demo():
    """Run complete demo"""
    print("🚀 ResNet Implementation - Complete Demo")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_model_creation()
        demo_configuration()
        demo_logging()
        demo_evaluation()
        demo_web_interface()
        
        print("\n🎉 Complete Demo Summary")
        print("=" * 60)
        print("✅ Model creation and testing")
        print("✅ Configuration management")
        print("✅ Logging and experiment tracking")
        print("✅ Evaluation and visualization")
        print("✅ Web interface setup")
        
        print("\n🚀 Next Steps:")
        print("1. Run training: python train.py")
        print("2. Start web interface: streamlit run app.py")
        print("3. Run tests: python test_resnet.py")
        print("4. Check configurations in ./configs/")
        
        print("\n📚 Documentation:")
        print("- README.md: Complete documentation")
        print("- configs/: Example configurations")
        print("- tests/: Unit tests")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Please check your installation and try again.")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='ResNet Implementation Demo')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'models', 'training', 'config', 'logging', 'evaluation', 'web'],
                       help='Which demo to run')
    
    args = parser.parse_args()
    
    if args.demo == 'all':
        run_full_demo()
    elif args.demo == 'models':
        demo_model_creation()
    elif args.demo == 'training':
        demo_training()
    elif args.demo == 'config':
        demo_configuration()
    elif args.demo == 'logging':
        demo_logging()
    elif args.demo == 'evaluation':
        demo_evaluation()
    elif args.demo == 'web':
        demo_web_interface()
    
    print("\n🎯 Demo completed successfully!")


if __name__ == "__main__":
    main()
