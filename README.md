# Complaint Classification using NLP (RoBERTa)

## Problem
Manual complaint categorization takes hours.

## Solution
Built a Transformer-based NLP model using RoBERTa to classify customer complaints automatically.

## Dataset
200K+ customer complaints.

## Model Architecture
RoBERTa Transformer + Ensemble logic for critical labels.

## Performance
F1 Score: 92%

## Tech Stack
Python
PyTorch
HuggingFace Transformers
Pandas
Scikit-learn

## How to Run

Install libraries:

pip install -r requirements.txt

Train model:

python model_training.py

Run prediction on future data:

python predict_future_data.py
