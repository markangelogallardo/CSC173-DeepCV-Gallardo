
# Evaluating Convolutional Neural Networks (CNNs) performance given different Data Augmentation Applications
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Mark Angelo L. Gallardo, 2022-182
**Semester:** AY 2025-2026 Sem 1 
[![Python](https://img.shields.io/badge/Python-3.13.11-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org)

## Abstract
Robustness of Lightweight Models and comparing their audio classification capability is what I aim to conduct here. Robustness of these models are important given that the environment they will be deployed at would have to deal with background noises that might still leak through for whatever classification task they will be assigned to, I wanted to focus in the agricultural setting. To do this, I utilized the ESC-50 dataset that has 50 different classes. The idea with this dataset was to split it to my 3 custom classes (resonant, damp, common) using 7 classes in the dataset, inspiration rooted from our study, and compare models in their capacity to evaluate these custom classes given augmented training data. I used 5 lightweigth models (>30mb) and trained them on 4 different augmentation setups (base/no augmentation, audio-only augmentation, spectrogram-only augmentation, and hybrid augmentation). They were validated and tested using the 5-fold cross validation as setup similar to the ESC-50 dataset. The results showed only 1 model achieving 100% accuracy in all 5 different fold setups but others also performed >90% in Accuracy, Precision, Recall. This project, past it being a personal project, contributes in evaluating different models trained with different augmentation techniques involving audio classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
&nbsp;&nbsp;&nbsp;&nbsp; Heavy background noise is rampant in the agricultural setting, sound emitted by livestock, machinery for processing crop, different weather conditions to name a few, and is much more apparent in the Philippines where the limited space for processing agricultural goods means overlapping sound profiles.    
&nbsp;&nbsp;&nbsp;&nbsp; The problem then is that in order for real-world application of detection systems in these settings, the discernment capability of AI models should be tested in datasets that replicate these scenarios.  
&nbsp;&nbsp;&nbsp;&nbsp; Notably on discerining between resonant and damped noises, add the fact as well that there are other common sounds in farms, and that there are sounds that act as background noise  
&nbsp;&nbsp;&nbsp;&nbsp; The project then aims to test the capability of different CNN models given different data availability environments in detecting 3 distinct classes: Resonant, Damp, and Common. Background noise will also be injected to the training dataset which are the following sound of : rain falling on metal roofing, and crickets chirping  
&nbsp;&nbsp;&nbsp;&nbsp; Resonant (glass_breaking, can_opening) which are sounds that sound "sharp" and unmuffled, damp (door_wood_knock, footsteps) which are sound that sounds "flat" and muffled, and common (hen, engine, rooster) which are sounds common in an agricultural setting. I just wanted to simulate the resonant-like sound produced by non-cracked eggs and damp-like sound produced by cracked eggs as well as the audio associated with the agricultural setting. The background noise was selected as well on the specific setting

### Objectives
- Split the 7 classes: glass_breaking, can_opening, door_wood_knock, footsteps, hen, engine, rooster of ESC50-Dataset to resonant (glass_breaking, can_opening), damp (door_wood_knock, footsteps) , and common (hen, engine, rooster)
- Apply data augmentation techniques to the training dataset and split them into the following groups:  
    - No data augmentation
    - Audio augmentation only
    - Spectrogram Augmentation only
    - Audio augmentation then Spectrogram augmentation 
- Train multiple pre-trained computer vision models using the different training dataset grouping to discern between 3 classes [resonant, damp, common] that have been split based on their spectral morphology.
- Validate their training per epoch using 5-fold Cross Validations
- Evaluate and compare their performance based on the following metrics:
    - Accuracy
    - Precision
    - Recall
    - F-1 Score

## Related Work
- [An evaluation of lightweight deep learning techniques in medical imaging for high precision COVID-19 diagnostics](https://www.sciencedirect.com/science/article/pii/S2772442522000417)

## Methodology
### Dataset
- Source: ESC-50 dataset
- Split: 80/20 split (4 folds to 1 fold) 
- Preprocessing: Time Stretch, Pitch Shift, Noise Injection, Frequency Masking, Time Masking, resizing to 224x224

### Architecture
MobileNetV3:  

![Model Diagram](images/mobilenetv3.png)
- Backbone: Mobile Inverted Bottleneck blocks (MBConv) with Squeeze-and-Excitation and Hard-Swish activation.
- Head: Global Average Pooling, then two Fully Connected layers with Hard-Swish and Dropout activation functions.  

MobileNetV2:  
![Model Diagram](images/mobilenetv2.png)
- Backbone: Inverted Residual blocks with linear bottlenecks and ReLU6 activation.
- Head: 1x1 Convolution, Global Average Pooling, Dropout, and a single Fully Connected layer.  

ShuffleNetV2:
![Model Diagram](images/shufflenetv2.png)
- Backbone: Shuffle Units with Channel Shuffle and depthwise separable convolutions.
- Head: 1x1 Convolution Global Average Pooling, and a single Fully Connected layer.  

EfficientNet:
![Model Diagram](images/efficientNet.png)
- Backbone: Mobile Inverted bottlenecks with Squeeze-and-Excitation
- Head: Global Average Pooling, 1x1 Convolution, and Fully Connected Layers  

RegNet:
![Model Diagram](images/regnet.png)
- Backbone: Quantized linear stack of Residual blocks with Group Convolutions and Squeeze-and-Excitation.
- Head: Global Average Pooling, a single Fully Connected layer.

Hyperparameters: Table below for all models

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 0.01 |
| Epochs | 50 |
| Optimizer | Adaptive Moment Estimation (Adam) |

### Training Code Snippet
train.py excerpt
```model = get_model(MODEL_NAME, num_classes=50)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Track best metrics for this fold
    best_metrics = {
        'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0
    }
    early_stopping_counter = 0

    # Train Loop
    for epoch in range(EPOCHS):
        
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in tqdm(train_loader, desc=f"Fold {fold_idx} Ep {epoch+1}/{EPOCHS}", leave=False):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        ep_train_loss = running_loss / len(train_loader)
        ep_train_acc = 100 * correct_train / total_train
```


## Experiments & Results
### Metrics and results after running test
| Architecture    | Data Augmentation        | Val_Accuracy | Val_Precision | Val_Recall | Val_F1 |
|-----------------|--------------------------|--------------|---------------|------------|--------|
| **mobilenet_v3**    | **Audio Augmentation**       | **100**          | **100**           | **100**        | **100**    |
| mobilenet_v2    | Audio Augmentation       | 98.21        | 98.41         | 98.21      | 98.21  |
| mobilenet_v2    | No Augmentation          | 98.21        | 98.41         | 98.21      | 98.21  |
| mobilenet_v2    | Spectrogram Augmentation | 98.21        | 98.41         | 98.21      | 98.21  |
| efficientnet_b0 | Hybrid Augmentation      | 98.21        | 98.41         | 98.21      | 98.21  |
| efficientnet_b0 | No Augmentation          | 98.21        | 98.41         | 98.21      | 98.21  |
| efficientnet_b0 | Spectrogram Augmentation | 98.21        | 98.41         | 98.21      | 98.21  |
| mobilenet_v3    | Hybrid Augmentation      | 98.21        | 98.41         | 98.21      | 98.21  |
| mobilenet_v3    | Spectrogram Augmentation | 98.21        | 98.41         | 98.21      | 98.21  |
| regnet_y_800mf  | Audio Augmentation       | 98.21        | 98.41         | 98.21      | 98.21  |
| regnet_y_800mf  | Hybrid Augmentation      | 98.21        | 98.41         | 98.21      | 98.21  |
| regnet_y_800mf  | Spectrogram Augmentation | 98.21        | 98.41         | 98.21      | 98.21  |
| shufflenet_v2   | No Augmentation          | 98.21        | 98.41         | 98.21      | 98.21  |
| mobilenet_v2    | Hybrid Augmentation      | 96.43        | 97.14         | 96.43      | 96.51  |
| regnet_y_800mf  | No Augmentation          | 96.43        | 96.83         | 96.43      | 96.28  |
| shufflenet_v2   | Audio Augmentation       | 96.43        | 97.14         | 96.43      | 96.37  |
| shufflenet_v2   | Hybrid Augmentation      | 96.43        | 96.83         | 96.43      | 96.41  |
| shufflenet_v2   | Spectrogram Augmentation | 96.43        | 97.14         | 96.43      | 96.51  |
| efficientnet_b0 | Audio Augmentation       | 94.64        | 96.1          | 94.64      | 94.45  |
| mobilenet_v3    | No Augmentation          | 94.64        | 95.36         | 94.64      | 94.72  |

Loss and Validation Curves are include in the folder for a specific model [here](saved_models)

### Demo
[CSC173_Gallardo_Final](https://youtu.be/U-9ciUagOXE)

## Discussion
- Strengths: Every model generally did well in detecting the 3 different classifiers
- Limitations: It is based on a dataset that already isn't a lot and data_augmentation is being applied
- Insights: Data Augmentation performance gains could've been better quantified with a bigger validation set. 

## Ethical Considerations
- Bias: Dataset limitation means only specific classes selected from ESC50 dataset might be recognized
- Privacy: No faces in training data, from publicly available dataset that has been selected to have no human voices

## Conclusion
The project highlights the potential of data augmentation in the improving the performance of lightweight models. In the future, this project could extend to more models being compared. Increase of the validation test could also give a more representative data. Lastly, deploy of these model to IoT use cases using mini computers such as Rasberry Pi's could make the system just be in an all in one package.


## Installation
1. Clone repo: `https://github.com/markangelogallardo/CSC173-DeepCV-Gallardo`
2. Install deps: `conda env create -f environment.yml`
3. Utilize weights: See `saved_models/` to use a specific model 

[**environment.yml:**](environment.yml)

## References
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks ](https://arxiv.org/abs/1905.11946)
- [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
- [ESC-50 Dataset LInk](https://www.researchgate.net/publication/305854186_ESC_Dataset_for_Environmental_Sound_Classification)

## GitHub Pages
View this project site: [https://github.com/markangelogallardo/CSC173-DeepCV-Gallardo](https://github.com/markangelogallardo/CSC173-DeepCV-Gallardo) 
