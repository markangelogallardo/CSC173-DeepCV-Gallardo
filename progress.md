# CSC173 Deep Computer Vision Project Progress Report
**Student:** Mark Angelo L. Gallardo, 2022-0182  
**Date:** December 15, 2025   
**Repository:** [Click This](https://github.com/markangelogallardo/CSC173-DeepCV-Gallardo)  


## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | 280 images downloaded/preprocessed |
| Data Augmentation | ✅ Completed| Generated 2520 images |
| Initial Training | ⏳ In Progress | Currently training with 280 no augmentation dataset (baseline) |
| Baseline Evaluation | ⏳ Not Started | Training ongoing |
| Model Fine-tuning | ⏳ Not Started | Planned for tomorrow |

## 1. Dataset Progress
- **Total images:** 2520 (280 no augmentation, 560 audio augmentation, 560 spectrogram augmentation, 1120 hybrid ) 
- **Train/Val/Test split:** 80/20 split (Augmented Data not yet taken into accoung)
- **Classes implemented:** Common, Resonant, Damp
- **Preprocessing applied:** Time Stretch, Pitch Shift, Noise Injection, Frequency Masking, Time Masking

**Sample data preview:**
![Dataset Sample](images/dataset_sample.png)

## 2. Training Progress

**Training Curves (so far)**
![Loss Curve](images/loss_curve.png)
![mAP Curve](images/map_curve.png)

**Current Metrics:**
| Metric | Train | Val |
|--------|-------|-----|
| Loss | [0.45] | [0.62] |
| mAP@0.5 | [78%] | [72%] |
| Precision | [0.81] | [0.75] |
| Recall | [0.73] | [0.68] |

## 3. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|-------|--------|------------|
| Normalizing Augmented Data  | ⏳ Ongoing | Implementing methods for ease of augmentation generation |
<!-- | CUDA out of memory | ✅ Fixed | Reduced batch_size from 32→16 |
| Class imbalance | ⏳ Ongoing | Added class weights to loss function |
| Slow validation | ⏳ Planned | Implement early stopping | -->

## 4. Next Steps (Before Final Submission)
- [ ] Complete training (50 more epochs)
- [ ] Hyperparameter tuning (learning rate, augmentations)
- [ ] Baseline comparison (vs. original pre-trained model)
- [ ] Record 5-min demo video
- [ ] Write complete README.md with results