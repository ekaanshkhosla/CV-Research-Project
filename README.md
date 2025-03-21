# 🚗 State Farm Distracted Driver Detection

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

This repository contains my solution for the **[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)** competition. The goal is to classify driver behaviors into 10 different categories to improve road safety.

---

## 🚦 Problem Statement

The dataset consists of images captured inside a car, where the task is to classify drivers into the following categories:

- `c0`: Safe driving  
- `c1`: Texting – right  
- `c2`: Talking on the phone – right  
- `c3`: Texting – left  
- `c4`: Talking on the phone – left  
- `c5`: Operating the radio  
- `c6`: Drinking  
- `c7`: Reaching behind  
- `c8`: Hair and makeup  
- `c9`: Talking to passenger  

---

## 📂 Project Structure

```
├── all_models/                                              # All Models used in Study
│   ├── Efficient_Net/                                       # Results of Efficient_Net
│   │   ├── CV_EfficientNet_V2_L_try1.db                     # Optuna Study db file   
│   │   ├── Efficient_Net_V2_L0806014                        # Logs
│   │   ├── Efficient_Net_V2_L.py                            # Optuna Study file
│   │   ├── efficientnet-with-data-augmentation.ipynb        # Efficient_Net with data augmentation
│   │   └── efficientnet-without-data-augmentation.ipynb     # Efficient_Net without data augmentation
│   ├── DinoV2/                                              # Results of DinoV2
│   │   ├── CV_DinoV2_optuna_large3.db                       # Optuna Study db file   
│   │   ├── DinoV2-with-augmentation.ipynb                   # DinoV2 with data augmentation
│   │   ├── DinoV2-without-augmentation.ipynb                # DinoV2 without data augmentation
│   │   ├── DinoV2_optuna_large.o797239                      # Logs
│   │   └── DinoV2_optuna_large.py                           # Optuna Study file
│   └── ConvNext/                                            # Results of ConvNext
│       ├── convNext_optuna.o801925                          # Logs
│       ├── convNext_optuna.py                               # Optuna Study file
│       ├── convNext_optuna_try4.db                          # Optuna Study db file  
│       ├── convnext-model-augmentation.ipynb                # convnext with data augmentation
│       └── convnext-model-without-augmentation.ipynb        # convnext without data augmentation
├── README.md                                                # README File
├── avg_Model.ipynb                                          # Ensemble Model
├── data_augmentation.ipynb                                  # Data Augmentation experiments
├── data_augmentation_apply.ipynb                            # Data Augmentation applied
└── train_valid_split.xlsx                                   # Train, Validation Split for Optuna Study
```
