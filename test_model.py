import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import precision_recall_fscore_support

from utils.load_dataset import DataAugmentedESC50Dataset
from utils.models import get_model

MODEL = "shufflenet_v2" # Change this to evaluate a different model architecture
MODELS_DIR = f"saved_models/{MODEL}" 
CSV_PATH = "data/metadata/log-mel_spectrograms_no_aug.csv"
DATA_DIR = "data/log-mel_spectrograms/no_aug"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
OUTPUT_REPORT_CSV = "model_evaluation_report.csv" 

def evaluate_on_fold(model, fold_num):
    dataset = DataAugmentedESC50Dataset(
        csv_file=CSV_PATH, root_dir=DATA_DIR, folds=[fold_num] 
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    return acc, precision, recall, f1

def main():
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    
    if not model_files:
        print(f"No .pth files found in {MODELS_DIR}")
        return

    print(f"Found {len(model_files)} models to evaluate.")
    
    summary_report = []

    for model_path in model_files:
        filename = os.path.basename(model_path)
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        
        arch_name = MODEL
        
        try:
            model = get_model(arch_name, num_classes=50)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
        except Exception as e:
            print(f"ERROR: Failed to load {filename}. Skipping. ({e})")
            continue

        fold_accuracies = []
        fold_precision = []
        fold_recall = []
        fold_f1s = []
        
        print(f"{'Fold':<5} | {'Acc':<8} | {'F1':<8}")
        print("-" * 30)
        
        for fold in [1, 2, 3, 4, 5]:
            acc, precision, recall, f1 = evaluate_on_fold(model, fold)
            fold_accuracies.append(acc)
            fold_precision.append(precision)
            fold_recall.append(recall)
            fold_f1s.append(f1)
            print(f"{fold:<5} | {acc:.2f}%   | {f1:.4f}")

        val_fold_idx = np.argmin(fold_accuracies) 
        val_fold_num = val_fold_idx + 1
        val_acc = fold_accuracies[val_fold_idx]
        val_precision = fold_precision[val_fold_idx]
        val_recall = fold_recall[val_fold_idx]
        val_f1 = fold_f1s[val_fold_idx]
        
        print(f"\n[DIAGNOSIS] Likely Validation Fold: {val_fold_num} (Acc: {val_acc:.2f}%)")
        
        summary_report.append({
            "Model_File": filename,
            "Architecture": arch_name,
            "Likely_Val_Fold": val_fold_num,
            "Val_Accuracy": val_acc,
            "Val_Precision": val_precision,
            "Val_Recall": val_recall,
            "Val_F1": val_f1,
            "Fold_1_Acc": fold_accuracies[0],
            "Fold_2_Acc": fold_accuracies[1],
            "Fold_3_Acc": fold_accuracies[2],
            "Fold_4_Acc": fold_accuracies[3],
            "Fold_5_Acc": fold_accuracies[4]
        })

    if summary_report:
        df = pd.DataFrame(summary_report)
        
        file_exists = os.path.isfile(OUTPUT_REPORT_CSV)
        
        df.to_csv(OUTPUT_REPORT_CSV, mode='a', header=not file_exists, index=False)
        
        print(f"\n{'='*60}")
        print(f"DONE. New results appended to: {OUTPUT_REPORT_CSV}")
        print(df[["Model_File", "Likely_Val_Fold", "Val_Accuracy"]])
if __name__ == "__main__":
    main()