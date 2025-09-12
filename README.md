# Garlic Leaf Disease Identification System

A garlic leaf disease identification system based on the improved ResNet18, which realizes the automatic classification and identification of garlic leaf diseases through deep learning technology.

## Project Overview

This project aims to construct an efficient garlic leaf disease identification model using deep learning technology. By leveraging the improved ResNet18 network architecture combined with attention mechanisms (CBAM and Triplet Attention), it achieves automatic identification of common garlic leaf diseases, providing support for early detection and diagnosis of diseases in agricultural production.

## Project Structure

```plaintext
Jingrun-Kan-Laixiang-Xu-garlic-leaf-disease-identification/
├── train_MyResnet18.py        # Main script for model training
├── MyResBet18.py              # Definition of the improved ResNet18 model
├── Triplet.py                 # Implementation of the Triplet Attention module
├── CBAM.py                    # Implementation of the CBAM attention mechanism
├── Pconv.py                   # Implementation of partial convolution
├── LeafDataset.py             # Dataset loading class
├── inference.py               # Model inference script
└── split_dataset/             # Dataset directory
    ├── train/                 # Training set
    ├── val/                   # Validation set
    └── test/                  # Test set
```

## Core Technologies

1. **Improved ResNet18**: The original ResNet18 is enhanced by integrating attention mechanisms.
2. Attention Mechanisms
   - CBAM (Convolutional Block Attention Module)
   - Triplet Attention
3. **Partial Convolution**: Reduces computational load and improves model efficiency.
4. **Transfer Learning**: Uses pre-trained ResNet18 weights as initial values.

## Recommended Environment

- Python 3.9
- PyTorch == 2.3.1
- torchvision == 0.18.1

## Dataset Preparation

The dataset should be organized in the following structure:

```plaintext
split_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

Each category folder contains garlic leaf images corresponding to that category.

## Model Training

Use the `train_MyResnet18.py` script for model training:

```bash
python train_MyResnet18.py
```

### Training Parameter Description

- Batch size: 4
- Number of training epochs: 100
- Image size: 224x224
- Optimizer: Adam
- Learning rate: Adopts a variable learning rate strategy
- Data augmentation: Includes Resize, ToTensor, and Normalize

During the training process, model weights will be automatically saved. After training, the model performance will be evaluated on the test set.

If you need to adjust hyperparameters, please modify the relevant parameters in `train_MyResnet18.py` manually.

## Model Inference

Use the provided inference script to make predictions on new images:

### Single Image Prediction

```bash
python inference.py --model_path myResNet18_bs32_ep100_224.pth --image_path .\split_dataset\test\Blight\20.jpg 
```

### Batch Image Prediction

```bash
python inference.py --model_path myResNet18_bs32_ep100_224.pth --image_dir test_images/
```

The inference script will output the predicted category and the corresponding confidence level.

## Performance Evaluation

After the training is completed, the model will be evaluated on the test set, and key metrics such as accuracy will be output. Users can also extend the evaluation metrics in `train_function.py` as needed.