# === Imports ===
from transformers import BertTokenizer, BertForSequenceClassification
#BertForSequenceClassification = pretrained BERT model for text classification tasks with a classification head added in its architecture
#BertTokenizer = tokenizer that processes raw text to tokens, an input format that BERT can understand

import torch #the machine learning framework used (pytorch)
from torch.utils.data import DataLoader, Dataset#tools for dataset handling and batching
from torch.optim import AdamW 
#adam is an algorithm for optimization in gradient decent. it uses gradient decent with momentum and RMSP (Root Mean Square Propogation) algorithms 
#gradient descent with momentum removes sudden changes in parameter values, smoothing it and fastens training
#RMSP adapts the learning rate for each parameter based on previous gradients

from torch.optim.lr_scheduler import ReduceLROnPlateau #learning rate scheduler

from sklearn.model_selection import StratifiedKFold
#library for making stratified k-cross validation, stratifying the fold ensures that proportion of the class labels in each split dataset is consistent with their proportion in the original dataset
#1/3 psychology labels, 1/3 sociology labels, and 1/3 political science labels proportion distribution

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix #the evaluation metrics used to measure performance
import pandas as pd s#library for data manipulation and analysis
import numpy as np #library for numerical operations in python
import matplotlib.pyplot as plt #library for general-purpose plotting
import seaborn as sns #visualization library for statistical plots, used for creating heatmap visualization part of the confusion matrix

# === Dataset loading ===
data = pd.read_csv('psycho-polsci-astro.csv') #loads the abstracts dataset to a pandas dataframe, which organizes the data into rows and columns. the columns are based on the csv which are "abstract" and "labels" columns
texts = data['abstract'].tolist()  #extracts "abstract" column from the dataframe and converts it into a entry list storing them in the variable "texts"
labels = data['label'].tolist()  #extracts "labels" column from the dataframe and converts it into a entry list storing them in the variable "labels"

# === String labels to numeric labels ===
label_map = {label: idx for idx, label in enumerate(set(labels))} #creates a dictionary that maps each label (Psychology, Political Science, Sociology) to a unique index (0, 1, 2)
numeric_labels = [label_map[label] for label in labels] #converts all labels in "labels" to to its numeric counterpart in "label_map", putting it in "numeric_labels" list

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #initializes a tokenizer from a pretrained BERT model

# === PyTorch dataset class ===
class AbstractDataset(Dataset):
    """this custom dataset class prepares the data and structures it for model training and evaluation"""
    
    def __init__(self, texts, labels, tokenizer, max_len): #init function, with parameters below:
        self.texts = texts #array of input texts
        self.labels = labels #array of numbers as label of texts
        self.tokenizer = tokenizer #tokenizer for converting text to tokens
        self.max_len = max_len #the set maximum inout sequence length of tokens 

    def __len__(self):
        return len(self.texts) #this method returns number of samples in the dataset aka length of the dataset

    def __getitem__(self, idx): #this method retrieves a specific text and its corresponding label, tokenizes the text and makes the sequence lengths consistent through padding and truncation for model input
        text = self.texts[idx] #fetches the text at the given index
        label = self.labels[idx] #fetches the corresponding label at the given index

        encoding = self.tokenizer(
            text, #input text
            max_length=self.max_len, #defines maximum length of text that is tokenzized
            padding='max_length', #padding makes sure that all texts that are inputted are the same length regardless of actual abstract text length
            truncation=True, #truncates texts if it is too long
            return_tensors="pt", #return format of tokenized text are tensors in PyTorch format
        )

        return { #returns tokenized text and the corresponding label as tensors (the models and loss functions expect inputs to be all tensors)
            'input_ids': encoding['input_ids'].squeeze(0),  #returns token IDs as a tensor
            'attention_mask': encoding['attention_mask'].squeeze(0),  #returns attention mask as a tensor, attention masks indicates which tokens in the input sequence are actual tokens (1) and which are padding tokens (0)
            'labels': torch.tensor(label, dtype=torch.long)  #converts the label into a PyTorch tensor before
        }

# === Confusion Matrix ===
def plot_confusion_matrix(cm, classes, title, filename):
    """function for plotting the confusion matrix with matplotlib for basic plotting and seaborn for heatmap element"""
    plt.figure(figsize=(10, 8))  #the size of the plot
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  #normalize confusion matrix values by row totals
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)  #generate the heatmap with seaborne
    #cm_normalized: normalized confusion matrix used to generate the heatmap
    #annot=True: enables annotations, each cell in the heatmap will display a value
    #fmt='.2f': annotation values are floats with 2 decimal places
    #cmap='Blues': heatmap color is shades of blue
    #xticklabels=classes, yticklabels=classes: sets labeling of x and y axis
    
    plt.title(title)  #a title to the plot
    plt.ylabel('True Labels')  #label the y-axis
    plt.xlabel('Predicted Labels')  #label the x-axis
    plt.tight_layout()  #adjusts layout to prevent overlapping elements
    plt.savefig(filename)  #saves the confusion matrix as an image file
    plt.close()  #close the plot to free memory

# === K-Fold Cross-Validation Setup ===
k_folds = 5 #the amount of folds for K-Fold cross-validation

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42) 
#initiates stratified K-fold cross-validation, dataset is split into 5 sub datasets
#shuffles the data to ensure randomness and avoid bias due to the order of the dataset
#random_state=42 is the fixed seed for the random number generator, ensuring that the shuffling and splitting of dataset are the same across multiple runs

fold_results = [] #empty list to store performance metrics at the result for each fold
aggregate_cm = np.zeros((len(label_map), len(label_map)))  #initialize a zero matrix for the aggregated confusion matrices across folds

# === Hyperparameters ===
batch_size = 8  #number of abstract text samples in a batch during training and validation, a batch is a subset of the dataset used to train the model in one forward and backward pass, the model processes this amount of text abstracts at a time during training or evaluation
epochs = 50  #maximum number of epochs to train the model, if early stopper is implemented it will rarely reach a high number of epoch
max_len = 512  #maximum sequence length for tokenization
early_stopping_patience = 3  #number of epochs to wait for improvement before stopping training early
scheduler_patience = 2  #number of epochs to wait before reducing the learning rate when loss value stagnates
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #makes sure GPU is used if available for faster training and uses CPU otherwise

# === Start K-Fold Cross-Validation ===
with open("metrics_logs_bert2.txt", "w") as log_file:  #opens log file in writing mode to the variable log_file for writing performance metrics purposes
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
                    preds = torch.argmax(logits, dim=-1) #

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
