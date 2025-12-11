
# CSC173 Deep Computer Vision Project Proposal
**Student:** Mark Angelo L. Gallardo, 2022-0182  
**Date:** December 11, 2025

## 1. Project Title 
Evaluating Convolutional Neural Networks (CNNs) performance in Audio Anomaly Detection in High-Noise Agricultural Environments

## 2. Problem Statement
&nbsp;&nbsp;&nbsp;&nbsp; Heavy background noise is rampant in the agricultural setting, sound emitted by livestock, machinery for processing crop, different weather conditions to name a few, and is much more apparent in the Philippines where the limited space for processing agricultural goods means overlapping sound profiles.  
&nbsp;&nbsp;&nbsp;&nbsp; The problem then is that in order for real-world application of detection systems in these settings, the discernment capability of AI models should be tested in datasets that replicate these scenarios. 
&nbsp;&nbsp;&nbsp;&nbsp; Notably on discerining between resonant and damped noises, add the fact as well that there are other common sounds in farms, and that there are sounds that act as background noise
&nbsp;&nbsp;&nbsp;&nbsp; The project then aims to test the capability of different CNN models given different data availability environments in detecting 3 distinct classes: Resonant, Damp, and Common. Background noise will also be injected to the training dataset which are the following sound of : rain falling on metal roofing, and crickets chirping

## 3. Objectives
- Apply data augmentation techniques to the training dataset and split them into the following groups:  
    - No data augmentation
    - Audio augmentation only
    - Spectrogram Augmentation only
    - Audio augmentation then Spectrogram augmentation 
- Train multiple pre-trained computer vision models using the different training dataset grouping to discern between 3 classes [resonant, damp, common] that have been split based on their spectral morphology.
- Validate their training per epoch
- Evaluate and compare their performance based on the following metrics:
    - Accuracy
    - Precision
    - Recall
    - F-1 Score

## 4. Dataset Plan
- Source: ESC-50, Youtube
- Classes:  
    - **Resonant** [Glass Breaking, Can Opening]
    - **Damp** [Door Knock, Footsteps]  
    - **Common** [Hen, Engine, Rooster] 
- Acquisition: 
    - Download ESC-50 dataset from github
    - Find rain and cricket sounds from Youtube

## 5. Technical Approach
- Architecture sketch
- Models (Pre-trained with ImageNet): MobileNetV3, RegNet MobileNetV2, ShuffleNetV2, EfficientNetB0
- Framework: PyTorch
- Hardware: NVIDIA GeForceRTX 4060 Laptop GPU, NVIDIA GeForceRTX 3050Ti Laptop GPU

## 6. Expected Challenges & Mitigations
Challenge 1: Data Scarcity  
- Mitigation: Data Augmentation by using 2 Data augmentation groups and utilizing pre-trained models
    - Audio-Domain: Waveform Gaussian noise injection and time-shifting to simulate sensor variability.

    - Feature-Domain: SpecAugment (Frequency and Time Masking) to force the model to learn varying spectral features

    - Transfer Learning: Utilization of ImageNet-pretrained weights with frozen backbones, ensuring the model relies on robust, pre-learned feature extractors rather than learning from scratch on a small dataset.

Challenge 2: Multiple Models need to be trained  
- Mitigation: Utilzie CUDA that is present in both of the researchers laptop, train concurrently the models 