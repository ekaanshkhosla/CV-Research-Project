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
├── all_models/
│   ├── Efficient_Net/
│   │   ├── CV_EfficientNet_V2_L_try1.db
│   │   ├── Efficient_Net_V2_L0806014
│   │   ├── Efficient_Net_V2_L.py
│   │   ├── efficientnet-with-data-augmentation.ipynb
│   │   └── efficientnet-without-data-augmentation.ipynb
│   ├── DinoV2/
│   │   ├── CV_DinoV2_optuna_large3.db
│   │   ├── DinoV2-with-augmentation.ipynb
│   │   ├── DinoV2-without-augmentation.ipynb
│   │   ├── DinoV2_optuna_large.o797239
│   │   └── DinoV2_optuna_large.py
│   └── ConvNext/
│       ├── convNext_optuna.o801925
│       ├── convNext_optuna.py
│       ├── convNext_optuna_try4.db
│       ├── convnext-model-augmentation.ipynb
│       └── convnext-model-without-augmentation.ipynb
├── README.md
├── avg_Model.ipynb
├── data_augmentation.ipynb
├── data_augmentation_apply.ipynb
└── train_valid_split.xlsx
