import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset loading
data = pd.read_csv('psycho-polsci-astro.csv') # Loads the abstracts dataset using pandas read_csv
texts = data['abstract'].tolist()  # Puts all the abstracts in the csv into the list "text"
labels = data['label'].tolist()  # Puts all the labels in the csv into the list "labels"
label_map = {label: idx for idx, label in enumerate(set(labels))} # Creates a dictionary that maps each label (Psychology, Political Science, Sociology) to a unique index (0, 1, 2)
numeric_labels = [label_map[label] for label in labels]

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# PyTorch Dataset
class AbstractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], max_length=self.max_len, padding='max_length',
                                  truncation=True, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, filename):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Labels'); plt.xlabel('Predicted Labels'); plt.tight_layout()
    plt.savefig(filename); plt.close()

# Training and Validation
def train_and_validate(model, optimizer, scheduler, train_loader, val_loader, device, patience, epochs):
    best_loss, patience_ctr = float('inf'), 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device),
                            labels=batch['labels'].to(device))
            outputs.loss.backward(); optimizer.step()

        model.eval(); val_loss, val_preds, val_true = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device),
                                labels=batch['labels'].to(device))
                val_loss += outputs.loss.item()
                val_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                val_true.extend(batch['labels'].cpu().numpy())

        scheduler.step(val_loss)
        if val_loss < best_loss: best_loss, patience_ctr = val_loss, 0
        else: patience_ctr += 1
        if patience_ctr >= patience: break
    return val_preds, val_true

# Main
device, batch_size, max_len, epochs, patience = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 8, 128, 50, 3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results, aggregate_cm = [], np.zeros((len(label_map), len(label_map)))

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, numeric_labels)):
    train_data = AbstractDataset([texts[i] for i in train_idx], [numeric_labels[i] for i in train_idx], tokenizer, max_len)
    val_data = AbstractDataset([texts[i] for i in val_idx], [numeric_labels[i] for i in val_idx], tokenizer, max_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map)).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    val_preds, val_true = train_and_validate(model, optimizer, scheduler, train_loader, val_loader, device, patience, epochs)
    acc, prec, rec, f1 = accuracy_score(val_true, val_preds), *precision_recall_fscore_support(val_true, val_preds, average='weighted')[:3]
    fold_results.append((acc, prec, rec, f1)); aggregate_cm += confusion_matrix(val_true, val_preds)

plot_confusion_matrix(aggregate_cm, list(label_map.keys()), "aggregate_cm.jpeg")
print("Average Metrics:", np.mean(fold_results, axis=0))
