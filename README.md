Bi-Mamba: Efficient Bi-Directional State Space Models for Learned Image Compression

![Python](https://img.shields.io/badge/Python-3.10-blue) 

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) 

![License](https://img.shields.io/badge/License-MIT-green)

📌 Abstract

JPEG

Ballé2018

Cheng2020

Particularly strong performance is observed at bitrates below 0.5 bpp.

📈 Rate-Distortion Curves

Kodak RD Curve

Tecnick RD Curve

📁 Repository Structure

model.py # Core BiMamba architecture

train1.py # Training pipeline

test_final_proposed_safe.py # Evaluation script

abla_final.py # Ablation study

phd_work2_final/ │── Average_Results.csv │── Detailed_Results.csv │── Kodak/ │── Tecnick/

⚙️ Installation

conda create -n bimamba python=3.10

conda activate bimamba

pip install torch torchvision numpy matplotlib pandas tqdm
▶️ Training

python train1.py

🧪 Evaluation

python test_final_proposed_safe.py

🧠 Model Overview

The proposed BiMamba architecture consists of:

Hierarchical encoder with bidirectional State Space layers

Multi-scale latent representation

Hyperprior-based entropy modeling

Context-adaptive probability estimation

Rate-distortion optimized training objective

The architecture achieves global spatial awareness while maintaining linear computational complexity O(n).
📄 Citation

If you use this work in your research, please cite:

@article{bimamba2026,

title={Bi-Mamba: Efficient Bi-Directional State Space Models for Learned Image Compression},

author={Renuka Govindaraju and S. Vidhusha},

year={2026} }
License

This project is released under the MIT License.
👩‍🔬 Authors

Renuka Govindaraju

S. Vidhusha
