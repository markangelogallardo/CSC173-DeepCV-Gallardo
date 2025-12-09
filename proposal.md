
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
    - audio augmentation -> spectrogram augmentation 
- Train multiple pre-trained computer vision models using the different training dataset grouping to discern between 3 classes that have been split based on their spectral morphology.
- Validate their training per epoch
- Evaluate and compare their performance based on the following metrics:
    - Recall
    - Precision
    - Inference Latency
- Generate a confusion matrix to which classes the models get confused


## 4. Dataset Plan
- Source: ESC-50, Youtube
- Classes:  
    - **Resonant** [Glass Breaking, Can Opening]
    - **Damp** [Door Knock, Footsteps]  
    - **Interference** [Hen, Engine, Rooster] 
- Acquisition: 
    - Download ESC-50 dataset from github
    - Find rain and cricket sounds from Youtube

## 5. Technical Approach
- Architecture sketch
- Models (Pre-trained with ImageNet): ResNet18, Swin Transformer Tiny, MobileNetV2, ShuffleNetV2, EfficientNetB0
- Framework: PyTorch
- Hardware: NVIDIA GeForceRTX 4060 Laptop GPU, NVIDIA GeForceRTX 3050Ti Laptop GPU

## 6. Expected Challenges & Mitigations
Challenge 1: Environmental Signal Masking: Continuous background interference (Class 2) may spectrally overlap with the target transient signals (Classes 0 and 1), potentially causing the model to miss short-duration impulses.

- Mitigation: Robustness Stress-Testing. The evaluation phase will subject the trained models to high-intensity localized noise profiles (Kuliglig engine rumble and Yero rain noise) at varying Signal-to-Noise Ratios (SNR). This quantifies the model's ability to isolate transient features even when the noise floor is significantly elevated.

Challenge 2: Data Scarcity: The filtered subset of proxy classes results in a limited dataset (~240 samples), which risks overfitting

- Mitigation: Data Augmentation, particularly 2 variations which are:
    - Audio-Domain: Waveform Gaussian noise injection and time-shifting to simulate sensor variability.

    - Feature-Domain: SpecAugment (Frequency and Time Masking) to force the model to learn varying spectral features

    - Transfer Learning: Utilization of ImageNet-pretrained weights with frozen backbones, ensuring the model relies on robust, pre-learned feature extractors rather than learning from scratch on a small dataset.