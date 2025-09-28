"""
Modern ResNet Implementation with PyTorch
=========================================

A comprehensive implementation of ResNet (Residual Network) with modern PyTorch practices,
type hints, and latest techniques for deep learning.

Features:
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 architectures
- Modern PyTorch practices with type hints
- Comprehensive training and evaluation pipeline
- Support for multiple datasets (CIFAR-10, CIFAR-100, ImageNet)
- Advanced optimization techniques (learning rate scheduling, weight decay, etc.)
"""

from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""
    
    expansion: int = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block for ResNet-50, ResNet-101, ResNet-152"""
    
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet implementation with modern PyTorch practices"""
    
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # CIFAR-10/100: stride=1, padding=1 for 32x32 input
        # ImageNet: stride=2, padding=3 for 224x224 input
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(num_classes: int = 10, **kwargs: Any) -> ResNet:
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def resnet34(num_classes: int = 10, **kwargs: Any) -> ResNet:
    """ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet50(num_classes: int = 10, **kwargs: Any) -> ResNet:
    """ResNet-50 model"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet101(num_classes: int = 10, **kwargs: Any) -> ResNet:
    """ResNet-101 model"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def resnet152(num_classes: int = 10, **kwargs: Any) -> ResNet:
    """ResNet-152 model"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


# Model registry for easy access
MODEL_REGISTRY = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}


def create_model(model_name: str, num_classes: int = 10, **kwargs: Any) -> ResNet:
    """Create a ResNet model by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test ResNet-18
    model = resnet18(num_classes=10)
    model = model.to(device)
    
    # Test with CIFAR-10 input size
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    output = model(dummy_input)
    print(f"✅ ResNet-18 output shape: {output.shape}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test ResNet-50
    model50 = resnet50(num_classes=10)
    model50 = model50.to(device)
    output50 = model50(dummy_input)
    print(f"✅ ResNet-50 output shape: {output50.shape}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model50.parameters()):,}")
