import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
FUTURE_FILE = r"C:\Users\515148627\Downloads\Recodes 22 to 31 Jan 2026 CC calls 1.xlsx"
OUTPUT_FILE = r"C:\Users\515148627\Downloads\DEBERTA1_PREDICTIONS_JAN_WK4.xlsx"
MODEL_DIR = r"C:\Users\515148627\OneDrive - GE Appliances\Dish Automation\Dish Codes Sept Onwards\Checkpoints\DeBERTa1_New_Masterfile_2025full"

TEXT_COLUMNS = ['TECH_CMNT_TXT', 'CNSMR_CPMLN_DSC', 'Part1', 'Part2', 'Part3', 'Part4', 'Part5']
MAX_LEN = 128
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL, TOKENIZER, LABEL ENCODER
# =========================
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "deberta1")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_deberta1")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_deberta1.pkl")

print("📥 Loading model, tokenizer, and label encoder...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH).to(DEVICE)
model.config.type_vocab_size = 1   # <--- ADD THIS
model.eval()

print("✅ Loaded successfully.")

# =========================
# DATASET CLASS
# =========================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
        }

# =========================
# LOAD FUTURE DATA
# =========================
print("📂 Reading input file...")
future_df = pd.read_excel(FUTURE_FILE)
future_df['text'] = future_df[TEXT_COLUMNS].fillna('').agg(' '.join, axis=1)

future_dataset = TextDataset(future_df['text'].tolist(), tokenizer, MAX_LEN)
future_loader = DataLoader(future_dataset, batch_size=BATCH_SIZE)

# =========================
# MAKE PREDICTIONS
# =========================
print("🔮 Generating predictions...")
all_preds = []
all_confidences = []

with torch.no_grad():
    for batch in future_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=None)

        probs = F.softmax(outputs.logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_confidences.extend(confs.cpu().numpy())

# =========================
# SAVE OUTPUT
# =========================
pred_labels = label_encoder.inverse_transform(all_preds)
future_df['Predicted_Problem'] = pred_labels
future_df['Prediction_Confidence'] = all_confidences

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
future_df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Predictions saved to: {OUTPUT_FILE}")
