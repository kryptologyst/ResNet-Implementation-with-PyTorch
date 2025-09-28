"""
Web UI for ResNet Model Inference and Visualization
==================================================

A modern web interface built with Streamlit for:
- Model inference on uploaded images
- Real-time predictions with confidence scores
- Grad-CAM visualization
- Model comparison
- Training progress monitoring
"""

import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Tuple, Optional
import cv2

from resnet import create_model, MODEL_REGISTRY
from evaluate import GradCAM, ModelEvaluator
from config import Config, ConfigManager


class ResNetWebApp:
    """Main web application class"""
    
    def __init__(self):
        self.setup_page_config()
        self.load_models()
        self.setup_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="ResNet Model Explorer",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_models(self):
        """Load available models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.model_configs = {}
        
        # CIFAR-10 class names
        self.cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # CIFAR-100 class names (simplified)
        self.cifar100_classes = [f"class_{i}" for i in range(100)]
    
    def setup_session_state(self):
        """Setup session state variables"""
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        if 'current_predictions' not in st.session_state:
            st.session_state.current_predictions = None
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
    
    def load_model(self, model_name: str, num_classes: int = 10, checkpoint_path: Optional[str] = None):
        """Load a specific model"""
        try:
            model = create_model(model_name, num_classes=num_classes)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                st.success(f"✅ Loaded {model_name} from {checkpoint_path}")
            else:
                st.warning(f"⚠️ Using untrained {model_name} (no checkpoint found)")
            
            model = model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            self.model_configs[model_name] = {
                'num_classes': num_classes,
                'checkpoint_path': checkpoint_path
            }
            
            return model
        except Exception as e:
            st.error(f"❌ Error loading model {model_name}: {str(e)}")
            return None
    
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (32, 32)) -> torch.Tensor:
        """Preprocess uploaded image for model inference"""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
        # Add batch dimension
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_image(self, model: torch.nn.Module, image_tensor: torch.Tensor) -> Dict[str, float]:
        """Make prediction on image tensor"""
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get top-5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, min(5, outputs.size(1)))
            
            predictions = {}
            for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
                predictions[f"top_{i+1}"] = {
                    'class': idx.item(),
                    'confidence': prob.item()
                }
            
            return predictions
    
    def create_gradcam_visualization(self, model: torch.nn.Module, image_tensor: torch.Tensor, 
                                   original_image: np.ndarray, target_layer: str = "layer4.1.conv2"):
        """Create Grad-CAM visualization"""
        try:
            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate_cam(image_tensor)
            
            # Resize CAM to match original image
            cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            
            # Overlay on original image
            overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
            
            return cam_resized, heatmap, overlay
        except Exception as e:
            st.error(f"Error creating Grad-CAM: {str(e)}")
            return None, None, None
    
    def render_sidebar(self):
        """Render sidebar with model selection and options"""
        st.sidebar.title("🧠 Model Configuration")
        
        # Model selection
        model_name = st.sidebar.selectbox(
            "Select Model Architecture",
            options=list(MODEL_REGISTRY.keys()),
            index=0
        )
        
        # Dataset selection
        dataset = st.sidebar.selectbox(
            "Select Dataset",
            options=["CIFAR-10", "CIFAR-100"],
            index=0
        )
        
        num_classes = 10 if dataset == "CIFAR-10" else 100
        
        # Checkpoint selection
        checkpoint_dir = "./checkpoints"
        available_checkpoints = []
        if os.path.exists(checkpoint_dir):
            available_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        
        checkpoint_path = None
        if available_checkpoints:
            checkpoint_file = st.sidebar.selectbox(
                "Select Checkpoint",
                options=["None"] + available_checkpoints,
                index=0
            )
            if checkpoint_file != "None":
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        else:
            st.sidebar.info("No checkpoints found. Using untrained model.")
        
        # Load model button
        if st.sidebar.button("Load Model", type="primary"):
            with st.spinner("Loading model..."):
                model = self.load_model(model_name, num_classes, checkpoint_path)
                if model is not None:
                    st.session_state.current_model = model
                    st.session_state.model_name = model_name
                    st.session_state.num_classes = num_classes
                    st.session_state.class_names = self.cifar10_classes if num_classes == 10 else self.cifar100_classes
        
        # Model info
        if st.session_state.current_model is not None:
            st.sidebar.success(f"✅ {st.session_state.model_name} loaded")
            param_count = sum(p.numel() for p in st.session_state.current_model.parameters())
            st.sidebar.metric("Parameters", f"{param_count:,}")
        
        # Visualization options
        st.sidebar.title("🎨 Visualization Options")
        show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
        show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
        
        return show_gradcam, show_confidence
    
    def render_main_content(self, show_gradcam: bool, show_confidence: bool):
        """Render main content area"""
        st.title("🧠 ResNet Model Explorer")
        st.markdown("Upload an image to see predictions and visualizations")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to classify"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            
            # Display original image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if st.session_state.current_model is not None:
                    st.subheader("🔮 Predictions")
                    
                    # Preprocess image
                    image_tensor = self.preprocess_image(image)
                    
                    # Make prediction
                    predictions = self.predict_image(st.session_state.current_model, image_tensor)
                    
                    # Display predictions
                    for i, (key, pred) in enumerate(predictions.items()):
                        class_name = st.session_state.class_names[pred['class']]
                        confidence = pred['confidence']
                        
                        # Create progress bar for confidence
                        st.write(f"**{i+1}. {class_name}**")
                        st.progress(confidence)
                        st.write(f"Confidence: {confidence:.3f}")
                    
                    st.session_state.current_predictions = predictions
                    
                    # Show Grad-CAM if requested
                    if show_gradcam:
                        st.subheader("🔥 Grad-CAM Visualization")
                        
                        # Convert PIL image to numpy array
                        original_array = np.array(image)
                        
                        # Create Grad-CAM visualization
                        cam, heatmap, overlay = self.create_gradcam_visualization(
                            st.session_state.current_model, image_tensor, original_array
                        )
                        
                        if cam is not None:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**Grad-CAM**")
                                st.image(cam, use_column_width=True)
                            
                            with col2:
                                st.write("**Heatmap**")
                                st.image(heatmap, use_column_width=True)
                            
                            with col3:
                                st.write("**Overlay**")
                                st.image(overlay, use_column_width=True)
                else:
                    st.warning("⚠️ Please load a model first using the sidebar")
    
    def render_model_comparison(self):
        """Render model comparison section"""
        st.title("⚖️ Model Comparison")
        
        if st.session_state.current_model is not None and st.session_state.current_predictions is not None:
            # Create comparison chart
            predictions = st.session_state.current_predictions
            
            classes = [st.session_state.class_names[pred['class']] for pred in predictions.values()]
            confidences = [pred['confidence'] for pred in predictions.values()]
            
            # Create bar chart
            fig = px.bar(
                x=classes,
                y=confidences,
                title="Prediction Confidence Scores",
                labels={'x': 'Class', 'y': 'Confidence'},
                color=confidences,
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create pie chart for top-3 predictions
            top3_classes = classes[:3]
            top3_confidences = confidences[:3]
            
            fig_pie = px.pie(
                values=top3_confidences,
                names=top3_classes,
                title="Top-3 Predictions Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_training_monitor(self):
        """Render training progress monitoring"""
        st.title("📊 Training Progress")
        
        # Check for training history files
        checkpoint_dir = "./checkpoints"
        history_files = []
        
        if os.path.exists(checkpoint_dir):
            history_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.json')]
        
        if history_files:
            selected_history = st.selectbox("Select Training History", history_files)
            
            if selected_history:
                history_path = os.path.join(checkpoint_dir, selected_history)
                
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    
                    # Create training curves
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Loss', 'Accuracy', 'Top-5 Accuracy', 'Learning Rate'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Loss curves
                    fig.add_trace(
                        go.Scatter(y=history['train_loss'], name='Train Loss', line=dict(color='blue')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(y=history['val_loss'], name='Val Loss', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    # Accuracy curves
                    fig.add_trace(
                        go.Scatter(y=history['train_acc'], name='Train Acc', line=dict(color='blue')),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(y=history['val_acc'], name='Val Acc', line=dict(color='red')),
                        row=1, col=2
                    )
                    
                    # Top-5 accuracy (if available)
                    if 'train_top5_acc' in history and 'val_top5_acc' in history:
                        fig.add_trace(
                            go.Scatter(y=history['train_top5_acc'], name='Train Top-5', line=dict(color='blue')),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(y=history['val_top5_acc'], name='Val Top-5', line=dict(color='red')),
                            row=2, col=1
                        )
                    
                    # Learning rate
                    fig.add_trace(
                        go.Scatter(y=history['learning_rate'], name='Learning Rate', line=dict(color='green')),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=800, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Best Train Acc", f"{max(history['train_acc']):.3f}")
                    with col2:
                        st.metric("Best Val Acc", f"{max(history['val_acc']):.3f}")
                    with col3:
                        st.metric("Final Train Loss", f"{history['train_loss'][-1]:.3f}")
                    with col4:
                        st.metric("Final Val Loss", f"{history['val_loss'][-1]:.3f}")
                
                except Exception as e:
                    st.error(f"Error loading training history: {str(e)}")
        else:
            st.info("No training history files found. Train a model first to see progress.")
    
    def run(self):
        """Run the web application"""
        # Render sidebar
        show_gradcam, show_confidence = self.render_sidebar()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🔮 Inference", "⚖️ Comparison", "📊 Training"])
        
        with tab1:
            self.render_main_content(show_gradcam, show_confidence)
        
        with tab2:
            self.render_model_comparison()
        
        with tab3:
            self.render_training_monitor()


def main():
    """Main function to run the Streamlit app"""
    app = ResNetWebApp()
    app.run()


if __name__ == "__main__":
    main()
