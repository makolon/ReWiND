#!/bin/bash
set -e

echo "Setting up ReWiND environment..."

# Create conda environment
echo "Creating conda environment 'rewind'..."
conda env create -f rewind.yml
conda activate rewind

# Install base requirements
echo "Installing base requirements..."
pip install -e .

# Setup metaworld_policy_training
echo "Setting up metaworld_policy_training..."
cd metaworld_policy_training


# Clone and install dependencies
echo "Installing mjrl..."
if [ ! -d "mjrl" ]; then
    git clone https://github.com/aravindr93/mjrl.git
fi
pip install -e mjrl

echo "Installing Metaworld..."
if [ ! -d "Metaworld" ]; then
    git clone https://github.com/sumedh7/Metaworld.git
fi
pip install -e Metaworld


# Install RL dependencies
echo "Installing RL dependencies..."
pip install stable-baselines3[extra]==1.8.0 --no-deps
pip install hydra-core==1.3.2

# Install PyTorch
echo "Installing PyTorch cu124..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Final installation
pip install -e .

echo "Setup complete! Activate the environment with: conda activate rewind"
