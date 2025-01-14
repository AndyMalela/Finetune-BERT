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

# Step 2: Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 3: Create a PyTorch Dataset Class
class AbstractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Custom function for plotting the confusion matrix
def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Step 4: K-Fold Cross-Validation
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_results = []
aggregate_cm = np.zeros((len(label_map), len(label_map)))  # To store the summed confusion matrix

# Hyperparameters
batch_size = 8
epochs = 50
max_len = 128
early_stopping_patience = 3
scheduler_patience = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start K-Fold Cross-Validation
with open("metrics_logs_bert2.txt", "w") as log_file:  # Open log file for writing
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, numeric_labels)):
        print(f"Fold {fold + 1}/{k_folds}")
        log_file.write(f"Fold {fold + 1}/{k_folds}\n")

        # Split data into training and validation sets
        train_texts = [texts[i] for i in train_idx]
        train_labels = [numeric_labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [numeric_labels[i] for i in val_idx]

        # Create PyTorch Datasets
        train_dataset = AbstractDataset(train_texts, train_labels, tokenizer, max_len)
        val_dataset = AbstractDataset(val_texts, val_labels, tokenizer, max_len)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Load BERT Model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
        model.to(device)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=0.1, verbose=True)

        # Early Stopping Variables
        best_val_loss = float('inf')
        patience_counter = 0

        # Training Loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            log_file.write(f"  Epoch {epoch + 1}/{epochs}\n")

            model.train()
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            average_epoch_loss = epoch_loss / len(train_loader)
            print(f"Training Loss after epoch {epoch + 1}: {average_epoch_loss}")
            log_file.write(f"    Training Loss: {average_epoch_loss:.4f}\n")

            # Validation Phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss  # Validation loss
                    val_loss += loss.item()

                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            average_val_loss = val_loss / len(val_loader)
            scheduler.step(average_val_loss)

            print(f"Validation Loss after epoch {epoch + 1}: {average_val_loss}")
            log_file.write(f"    Validation Loss: {average_val_loss:.4f}\n")

            # Early Stopping Check
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')
            else:
                patience_counter += 1
                print(f"Early stopping patience counter: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        # Calculate Metrics
        accuracy = accuracy_score(val_true, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='weighted')
        cm = confusion_matrix(val_true, val_preds)

        print(f"Fold {fold + 1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        log_file.write(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")

        # Add to aggregate confusion matrix
        aggregate_cm += cm

        # Append metrics for averaging
        fold_results.append((accuracy, precision, recall, f1))

    # Step 5: Normalize and Save the Aggregate Confusion Matrix
    classes = list(label_map.keys())
    plot_confusion_matrix(aggregate_cm, classes, "Aggregate Confusion Matrix (Percentage)", "aggregate_confusion_matrix_bert2.jpeg")

    # Step 6: Calculate and Save Average Metrics
    log_file.write("\nAverage Metrics Across All Folds:\n")
    if fold_results:
        avg_accuracy = np.mean([result[0] for result in fold_results])
        avg_precision = np.mean([result[1] for result in fold_results])
        avg_recall = np.mean([result[2] for result in fold_results])
        avg_f1 = np.mean([result[3] for result in fold_results])
    else:
        avg_accuracy = avg_precision = avg_recall = avg_f1 = 0.0  # Default to 0 if no metrics are available

    log_file.write(f"  Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}\n")

model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
