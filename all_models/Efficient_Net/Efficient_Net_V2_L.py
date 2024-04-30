import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import cv2
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import numpy as np
import optuna
from sklearn.metrics import f1_score
from optuna.exceptions import TrialPruned
import random

working_folder = os.path.abspath("")
image_dir = os.path.join(working_folder, "all_images")

train_df = 'train_split.csv'
val_df = 'valid_split.csv'

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 1]
        img_name = os.path.join(self.root_dir, f"{img_id}")
        image = Image.open(img_name).convert('RGB')
        label = int(self.annotations.iloc[index, 0][-1])  # Extract class number

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
def print_hyperparameters(image_size, batch_size, learning_rate, fc_units, dropout_rate, layer_freeze_upto):
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate:.6f}")
    print(f"Fully connected layer: {fc_units}")
    print(f"Dropout rate: {dropout_rate:.6f}")
    print(f"Layer Freeze Upto: {layer_freeze_upto}")
    
    
def define_model(layer_freeze_upto, fc_units, dropout_rate, num_classes):
    
    model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

    cutoff_reached = False
    for name, param in model.named_parameters():
        if not cutoff_reached:
            if name == layer_freeze_upto:
                cutoff_reached = True
            param.requires_grad = False
        else:
            param.requires_grad = True
        
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, fc_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(fc_units, num_classes),
    )
    
    return model


storage_url = "sqlite:///CV_EfficientNet_V2_L_try1.db"
study_name = "CV_EfficientNet_V2_L_try1"

study = optuna.create_study(study_name=study_name, storage=storage_url, direction="minimize", load_if_exists=True)

def objective(trial):
    
    # Hyperparameters
    image_size = trial.suggest_categorical('image_size', [128, 224, 256])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.001, log=True)
    fc_units = trial.suggest_int('fc_units', 64, 2048, step = 64)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.8, step = 0.05)
    layer_freeze_upto = trial.suggest_categorical('layer_freeze_upto', ['features.7.6.block.3.1.bias',
                                                                        'features.6.24.block.3.1.bias',
                                                                        'features.5.18.block.3.1.bias',
                                                                        'features.4.9.block.3.1.bias',
                                                                        'features.3.6.block.1.1.bias', 
                                                                        'features.2.6.block.1.1.bias',
                                                                        'features.1.3.block.0.1.bias',
                                                                        'features.0.1.bias'])
    
    num_classes = 10

    print("====================",f"Training of trial number:{trial.number}","====================")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print_hyperparameters(image_size, batch_size, learning_rate, fc_units, dropout_rate, layer_freeze_upto)

    
    train_dataset = CustomImageDataset(train_df, image_dir, transform=transform)
    val_dataset = CustomImageDataset(val_df, image_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = define_model(layer_freeze_upto, fc_units, dropout_rate, num_classes)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    patience = 10

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        min_multi_class_log_loss = study.best_trial.value
    else:
        min_multi_class_log_loss = 1000000000

    print("Best multi_class_log_loss on Validation data until now:", min_multi_class_log_loss)
    epochs_no_improve = 0
    trail_best = 1000000000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in (train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss/len(train_loader)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in (val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

        val_epoch_loss = val_running_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}')

        trial.report(val_epoch_loss, epoch)
        if trial.should_prune():
            print("Best multi_class_log_loss on Validation data until now:", min_multi_class_log_loss)
            raise TrialPruned()

        if(val_epoch_loss > 1):
            print("Very bad trail")
            print("Best multi_class_log_loss on Validation data until now:", min_multi_class_log_loss)
            break

        if val_epoch_loss < trail_best:
            trail_best = val_epoch_loss
            epochs_no_improve = 0
            print("Best multi_class_log_loss till now on the current trail on Validation data:", trail_best)
            if(val_epoch_loss < min_multi_class_log_loss):
                min_multi_class_log_loss = val_epoch_loss
                print("Best multi_class_log_loss on Validation data until now:", min_multi_class_log_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                print("Best multi_class_log_loss on Validation data until now:", min_multi_class_log_loss)
                break
    
    return min_multi_class_log_loss


completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
if(len(completed_trials) > 0):
    best_trial = study.best_trial
    print("Best trial's number: ", best_trial.number)
    print(f"Best score: {best_trial.value}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")
        
        
total_trails_to_run = 100

completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
num_trials_completed = len(completed_trials)

pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
num_pruned_trials = len(pruned_trials)

print(f"Number of trials completed: {num_trials_completed}")
print(f"Number of pruned trials: {num_pruned_trials}")
print(f"Total number of trails completed: {num_trials_completed + num_pruned_trials}")

trials_to_run = max(0, total_trails_to_run - (num_trials_completed + num_pruned_trials))
print(f"Number of trials to run: {trials_to_run}")


study.optimize(objective, trials_to_run)


best_trial = study.best_trial
print("Best trial's number: ", best_trial.number)
print(f"Best score: {best_trial.value}")
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")