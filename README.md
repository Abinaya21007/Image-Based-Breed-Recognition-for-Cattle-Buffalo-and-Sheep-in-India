# 🐄🐃🐑 Image-Based-Breed-Recognition-for-Cattle-Buffalo-and-Sheep-in-India

### Novel Architecture: Swin Transformer + CBAM (Intermediate Integration)

## Overview
  This project presents a deep learning-based Livestock Breed Recognition System capable of identifying breeds across three animal categories — Cattle (Cow), Buffalo, and Sheep — using a novel architecture that integrates Swin Transformer with CBAM (Convolutional Block Attention Module) at intermediate stages.Unlike prior works that apply attention mechanisms only at the end of the network, this system inserts CBAM between each Swin Transformer stage, enabling fine-grained feature extraction of breed-specific visual characteristics such as body structure, coat texture, horn shape, and facial features.

## Results

Animal Breeds Accuracy

Cattle (11 breeds) - 99.7%  

Buffalo (17 breeds) - 92.76%

Sheep (7 breeds) - 98.48%

## Novel Architecture

  Input Image → Patch Partition → Swin Stage 1 → **CBAM 1** → Swin Stage 2 → **CBAM 2** → Swin Stage 3 → **CBAM 3** → Swin Stage 4 → Global Average Pooling → Fully Connected Layer → Breed Output

## Why This is Novel:

  Standard works apply CBAM only at the end of the network
  This system injects CBAM between Swin Transformer stages
  Allows the attention module to refine features at every scale
  First application of this architecture for livestock breed recognition

## Repository Structure

livestock-breed-recognition/
│

├── cow/

│   ├── augment.py   

│   └── swin_cbam_cow.ipynb    

│
├── buffalo/

│   └── swin_cbam_buffalo.ipynb  
│
├── sheep/

│   └── swin_cbam_sheep.ipynb       
│
├── gradio_app/

│   └── app.py                      
│
└── README.md

## Getting Started

1. Clone the Repository

bashgit clone https://github.com/Abinaya21007/livestock-breed-recognition.git

cd livestock-breed-recognition

2. Install Dependencies

bashpip install torch torchvision timm gradio Pillow scikit-learn seaborn matplotlib

3. Run Training (Google Colab recommended)

Open the respective .ipynb notebook in Google Colab with GPU enabled.

4. Run Gradio Web App

  bashpython gradio_app/app.py

## 📓 Notebooks

 Sheep Breed Recognition [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iudWBl7UZmaiCJ9WoFqBPHVmzzXMEjyh?usp=sharing)

Cow Breed Recognition [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1lFzm3z3oGrv9UkeLuAjzKy6G-Ap6Hd5o?usp=sharing)

Buffalo Breed Recognition [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r1zkVxuXwXz3LRlOmeGItSi57A7oVOmF?usp=sharing) 
