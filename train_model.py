
# deberta_v3_final.py
# ================= FORCE CPU ONLY =================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
torch.cuda.is_available = lambda: False
torch.backends.cudnn.enabled = False
torch.set_default_device("cpu")
# =================================================
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from itertools import chain

# Download WordNet for synonym replacement
nltk.download('wordnet')

# =========================
# CONFIG
# =========================
TRAIN_FILE = r"data/train_data.xlsx"
FUTURE_FILE = r"data/future_data.xlsx"
OUTPUT_FILE = r"outputs/predictions.xlsx"
MODEL_DIR = r"models/deberta_model"

TEXT_COLUMNS = ['TECH_CMNT_TXT', 'CNSMR_CPMLN_DSC', 'Part1', 'Part2', 'Part3', 'Part4', 'Part5']
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
BATCH_SIZE = 32
ACCUMULATION_STEPS = 2
EPOCHS = 10
LR = 2e-5
DEVICE = torch.device("cpu")
EARLY_STOPPING_PATIENCE = 3

os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_deberta1.pkl")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_deberta1")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "deberta1")

# =========================
# DATA AUGMENTATION FUNCTIONS
# =========================
def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) == 0:
        return text
    new_words = [w for w in words if random.random() > p]
    return " ".join(new_words) if new_words else random.choice(words)

def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([w for w in words if wordnet.synsets(w)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for word in random_word_list:
        synonyms = wordnet.synsets(word)
        if synonyms:
            syn_words = set(chain.from_iterable([s.lemma_names() for s in synonyms]))
            syn_words.discard(word)
            if len(syn_words) > 0:
                new_words = [random.choice(list(syn_words)) if w == word else w for w in new_words]
                num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_words)

def augment_text(text):
    text = random_deletion(text, p=0.1)
    text = synonym_replacement(text, n=1)
    return text

# =========================
# DATASET CLASS
# =========================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment:
            text = augment_text(text)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# =========================
# TRAINING FUNCTION
# =========================
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs,
    device,
    accumulation_steps=1
):
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs}")

        # =========================
        # TRAIN
        # =========================
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / accumulation_steps
            loss.backward()

            total_train_loss += loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"✅ Training Loss: {avg_train_loss:.4f}")

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        print(
            f"📊 Validation Loss: {avg_val_loss:.4f} | "
            f"Accuracy: {val_accuracy:.4f}"
        )

        # =========================
        # EARLY STOPPING + SAVE BEST
        # =========================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            print(f"💾 Saving best model to {MODEL_SAVE_PATH} ...")
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(TOKENIZER_PATH)
            joblib.dump(label_encoder, LABEL_ENCODER_PATH)

            print("✅ Model, tokenizer, and label encoder saved.")
        else:
            patience_counter += 1
            print(
                f"⏳ No improvement. "
                f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

    return model


# =========================
# MAIN TRAIN / LOAD LOGIC
# =========================
if not os.path.exists(os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin")):
    print("📥 No existing model found. Training a new DeBERTa1 model...")

    # Load training data
    df = pd.read_excel(TRAIN_FILE)
    df['text'] = df[TEXT_COLUMNS].fillna('').agg(' '.join, axis=1)
    labels = df['Problem'].astype(str).values

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"✅ Label encoder saved to {LABEL_ENCODER_PATH}")

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), labels_encoded, test_size=0.2, random_state=42
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print(f"✅ Tokenizer saved to {TOKENIZER_PATH}")

    # Datasets and loaders
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN, augment=True)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_encoder.classes_)
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Train
    model = train_model(model, train_loader, val_loader, optimizer, scheduler, EPOCHS, DEVICE, ACCUMULATION_STEPS)

else:
    print("📥 Existing model found. Loading DeBERTa1 model, tokenizer, and label encoder...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH).to(DEVICE)

# =========================
# PREDICTIONS ON FUTURE DATA
# =========================
future_df = pd.read_excel(FUTURE_FILE)
future_df['text'] = future_df[TEXT_COLUMNS].fillna('').agg(' '.join, axis=1)
future_dataset = TextDataset(future_df['text'].tolist(), [0]*len(future_df), tokenizer, MAX_LEN)
future_loader = DataLoader(future_dataset, batch_size=BATCH_SIZE)

model.eval()
all_preds = []
all_confidences = []

with torch.no_grad():
    for batch in future_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_confidences.extend(confs.cpu().numpy())

# Decode labels
pred_labels = label_encoder.inverse_transform(all_preds)

# Save predictions
future_df['Predicted_Problem'] = pred_labels
future_df['Prediction_Confidence'] = all_confidences
future_df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Future predictions saved to {OUTPUT_FILE}")
