import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset 
from torch.cuda.amp import autocast, GradScaler
from utils.load_dataset import DataAugmentedESC50Dataset
from utils.models import get_model
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore", category=FutureWarning)

CLEAN_CSV_PATH = "data/metadata/log-mel_spectrograms_no_aug.csv"
CLEAN_SPEC_DIR = "data/log-mel_spectrograms/no_aug"

DATASET_NAME = "No_Augmentation" #change this to reflect dataset type: "No_Augmentation", "Audio_Augmentation", "Hybrid_Augmentation"

TYPE = "no"  # no, audio, hybrid
TRAIN_CSV_PATH = f"data/metadata/log-mel_spectrograms_{TYPE}_aug.csv"  
TRAIN_SPEC_DIR = f"data/log-mel_spectrograms/{TYPE}_aug" 

MODEL_NAME = "mobilenet_v3" 
BATCH_SIZE = 32 
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = f"saved_models/{MODEL_NAME}/"
RESULTS_CSV_PATH = "experiment_results.csv" 
os.makedirs(SAVE_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

EARLY_STOPPING_PATIENCE = 7   
EARLY_STOPPING_MIN_DELTA = 0.1 

def train_one_fold(fold_idx, train_folds, val_folds):
    print(f"\n{'='*20} Starting Fold {fold_idx} {'='*20}")
    
    primary_train_dataset = DataAugmentedESC50Dataset(
        csv_file=TRAIN_CSV_PATH, root_dir=TRAIN_SPEC_DIR, folds=train_folds
    )

    if DATASET_NAME == "No_Augmentation":
        print(f"[INFO] Dataset: No_Augmentation. Using single dataset source.")
        final_train_dataset = primary_train_dataset
    else:
        print(f"[INFO] Dataset: {DATASET_NAME}. Combining Augmented + Clean data.")
        
        clean_train_dataset = DataAugmentedESC50Dataset(
            csv_file=CLEAN_CSV_PATH, root_dir=CLEAN_SPEC_DIR, folds=train_folds
        )
        
        final_train_dataset = ConcatDataset([primary_train_dataset, clean_train_dataset])

    val_dataset = DataAugmentedESC50Dataset(
        csv_file=CLEAN_CSV_PATH, root_dir=CLEAN_SPEC_DIR, folds=val_folds
    )

    train_loader = DataLoader(
        final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, 
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    model = get_model(MODEL_NAME, num_classes=50)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_metrics = {
        'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0
    }
    early_stopping_counter = 0

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

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        ep_val_loss = running_val_loss / len(val_loader)
        ep_val_acc = 100 * correct_val / total_val

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )

        scheduler.step(ep_val_loss)

        fold_history['train_loss'].append(ep_train_loss)
        fold_history['train_acc'].append(ep_train_acc)
        fold_history['val_loss'].append(ep_val_loss)
        fold_history['val_acc'].append(ep_val_acc)

        print(f"   Ep {epoch+1}: Acc: {ep_val_acc:.2f}% | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        if ep_val_acc > (best_metrics['acc'] + EARLY_STOPPING_MIN_DELTA):
            best_metrics = {
                'acc': ep_val_acc,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100
            }
            early_stopping_counter = 0
            temp_path = os.path.join(SAVE_DIR, f"temp_{MODEL_NAME}_{DATASET_NAME}_fold{fold_idx}.pth")
            torch.save(model.state_dict(), temp_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break 
    
    return fold_history, best_metrics

def save_results_to_csv(avg_results):
    new_entry = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model': MODEL_NAME,
        'Dataset_Type': DATASET_NAME,
        'Avg_Accuracy': avg_results['acc'],
        'Avg_F1_Score': avg_results['f1'],
        'Avg_Precision': avg_results['precision'],
        'Avg_Recall': avg_results['recall'],
        'Std_Accuracy': avg_results['acc_std']
    }
    
    df_new = pd.DataFrame([new_entry])
    
    if not os.path.exists(RESULTS_CSV_PATH):
        df_new.to_csv(RESULTS_CSV_PATH, index=False)
    else:
        df_new.to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
        
    print(f"\n[INFO] Results saved to {RESULTS_CSV_PATH}")

def run_cross_validation():
    all_folds_history = []
    fold_results = [] 
    all_folds = [1, 2, 3, 4, 5]

    for i in range(5):
        val_folds = [all_folds[i]]
        train_folds = [x for x in all_folds if x not in val_folds]
        
        history, best_metrics = train_one_fold(i+1, train_folds, val_folds)
        all_folds_history.append(history)
        fold_results.append(best_metrics)
    
    accs = [r['acc'] for r in fold_results]
    precs = [r['precision'] for r in fold_results]
    recs = [r['recall'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]

    avg_acc = np.mean(accs)
    std_acc = np.std(accs)

    print("\n" + "="*40)
    print("FINAL CROSS VALIDATION RESULTS (5-FOLD)")
    print("="*40)
    
    print(f"{'Fold':<5} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 40)
    for i, res in enumerate(fold_results):
        print(f"{i+1:<5} {res['acc']:.2f}%   {res['f1']:.2f}%   {res['precision']:.2f}%   {res['recall']:.2f}%")
    print("-" * 40)

    print(f"AVG   {avg_acc:.2f}%   {np.mean(f1s):.2f}%   {np.mean(precs):.2f}%   {np.mean(recs):.2f}%")
    print("="*40)

    best_fold_idx = np.argmax(accs) + 1 
    best_fold_acc = accs[best_fold_idx - 1]
    
    print(f"\n[MODEL SELECTION] Best Fold: {best_fold_idx} (Acc: {best_fold_acc:.2f}%)")
    
    best_temp_path = os.path.join(SAVE_DIR, f"temp_{MODEL_NAME}_{DATASET_NAME}_fold{best_fold_idx}.pth")
    final_model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_{DATASET_NAME}_best.pth")
    
    if os.path.exists(best_temp_path):
        if os.path.exists(final_model_path):
            os.remove(final_model_path) 
        os.rename(best_temp_path, final_model_path)
        print(f"-> Saved winner to: {final_model_path}")
    
    print("[CLEANUP] Removing suboptimal fold checkpoints...")
    for i in range(1, 6):
        if i != best_fold_idx:
            temp_path = os.path.join(SAVE_DIR, f"temp_{MODEL_NAME}_{DATASET_NAME}_fold{i}.pth")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    save_results_to_csv({
        'acc': avg_acc,
        'f1': np.mean(f1s),
        'precision': np.mean(precs),
        'recall': np.mean(recs),
        'acc_std': std_acc
    })

    plot_average_curves(all_folds_history)

def plot_average_curves(all_histories):
    max_len = max([len(h['train_loss']) for h in all_histories])
    def pad_metric(metric_list, target_len):
        last_val = metric_list[-1]
        return metric_list + [last_val] * (target_len - len(metric_list))

    metrics = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    avg_data = {m: [] for m in metrics}

    for metric in metrics:
        matrix = []
        for h in all_histories:
            matrix.append(pad_metric(h[metric], max_len))
        avg_data[metric] = np.mean(matrix, axis=0)

    epochs = range(1, max_len + 1)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_data['train_loss'], label='Avg Train Loss', color='blue')
    plt.plot(epochs, avg_data['val_loss'], label='Avg Val Loss', color='orange')
    plt.title(f'{MODEL_NAME} - 5-Fold Avg Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_data['train_acc'], label='Avg Train Acc', color='blue')
    plt.plot(epochs, avg_data['val_acc'], label='Avg Val Acc', color='orange')
    plt.title(f'{MODEL_NAME} - 5-Fold Avg Accuracy')
    plt.legend(); plt.grid(True)
    
    save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_{DATASET_NAME}_5fold_average.png")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    run_cross_validation()