"""
Setup script for ResNet Implementation package
==============================================

Installation script for the ResNet implementation package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="resnet-implementation",
    version="1.0.0",
    author="AI Projects",
    author_email="ai.projects@example.com",
    description="A comprehensive ResNet implementation with modern PyTorch practices",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ai-projects/resnet-implementation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "web": [
            "streamlit>=1.15.0",
        ],
        "tracking": [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0",
        ],
        "optimization": [
            "optuna>=3.0.0",
            "ray[tune]>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resnet-train=train:main",
            "resnet-evaluate=evaluate:main",
            "resnet-app=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "configs/*.json"],
    },
    keywords="resnet, deep learning, pytorch, computer vision, image classification",
    project_urls={
        "Bug Reports": "https://github.com/ai-projects/resnet-implementation/issues",
        "Source": "https://github.com/ai-projects/resnet-implementation",
        "Documentation": "https://github.com/ai-projects/resnet-implementation/wiki",
    },
)
