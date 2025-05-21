# === Imports ===
import random # python rng library, for setting seed
from transformers import BertTokenizer, BertForSequenceClassification
#BertForSequenceClassification = pretrained BERT model for text classification tasks with a classification head added in its architecture
#BertTokenizer = tokenizer that processes raw text to tokens, an input format that BERT can understand

import torch #the machine learning framework used (pytorch)
from torch.utils.data import DataLoader, Dataset 
#tools for dataset handling and batching
#Dataset is a base class in PyTorch that is used to represent a dataset that allows to define how data samples (text and labels) are accessed and processed
#DataLoader is a utility that complements Dataset. In this code it wraps train_dataset and val_dataset instances providing batching and optionally shuffled data during training and validation

from torch.optim import AdamW 
#adamw is an algorithm for optimization in gradient decent. it uses gradient decent with momentum and RMSP (Root Mean Square Propogation) algorithms with weight decay.
#gradient descent with momentum removes sudden changes in parameter values, smoothing it and fastens training
#RMSP adapts the learning rate for each parameter based on previous gradients

from torch.optim.lr_scheduler import ReduceLROnPlateau 
#learning rate scheduler, adjusts the learning rate based on a monitored metric (here is validation loss)
#used to adjust the learning rated during training based on the performance of validation loss. If the loss does not improve after a certain number of epochs, learning rate is reduced t by a certain factor.

from sklearn.model_selection import StratifiedKFold
#library for making stratified k-cross validation, stratifying the fold ensures that proportion 
# of the class labels in each split dataset is consistent with their proportion in the original dataset
#1/3 psychology labels, 1/3 sociology labels, and 1/3 political science labels proportion distribution

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix #the evaluation metrics used to measure performance
import pandas as pd #library for data manipulation and analysis
import numpy as np #library for numerical operations in python
import matplotlib.pyplot as plt #library for general-purpose plotting
import seaborn as sns #visualization library for statistical plots, used for creating heatmap visualization part of the confusion matrix

# === Set the Seed for Various Reproducibility in Operations ===
seed = 69  #sets seed value, crucial to keep the same value across model training runs
random.seed(seed) #sets the seed for random number generator
np.random.seed(seed) #ensures reproducibility in NumPy operations
torch.manual_seed(seed) #controls seed for operations by CPU for deterministic results
torch.cuda.manual_seed_all(seed)  #controls seed for operations by PyTorch's CUDA GPU for deterministic results (if using GPU)

# === Dataset loading ===
data = pd.read_csv('psycho-polsci-socio.csv') #loads the abstracts dataset to a pandas dataframe, which organizes the data into rows and columns. the columns are based on the csv which are "abstract" and "labels" columns
texts = data['abstract'].tolist()  #extracts "abstract" column from the dataframe and converts it into a entry list storing them in the variable "texts"
labels = data['label'].tolist()  #extracts "labels" column from the dataframe and converts it into a entry list storing them in the variable "labels"

# === String labels to numeric labels ===
label_map = {label: idx for idx, label in enumerate(set(labels))} #creates a dictionary that maps each label (Psychology, Political Science, Sociology) to a unique index (0, 1, 2)
numeric_labels = [label_map[label] for label in labels] #converts all labels in "labels" to to its numeric counterpart in "label_map", putting it in "numeric_labels" list

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
#initializes a tokenizer from a pretrained BERT model, WordPiece tokenization algorithm is a part of here, this tokenizer ignores cased charaacters

# === PyTorch dataset class ===
class AbstractDataset(Dataset):
    """
    This custom dataset class subclasses 'Dataset' prepares the data and structures it for model training and evaluation
    It handles storing texts and labels, and tokenizing the text and returning the processed data (input_ids, attention_mask, labels) when accessed by an index
    """
    
    def __init__(self, texts, labels, tokenizer, max_len): #init function, with parameters below:
        self.texts = texts #array of input texts
        self.labels = labels #array of numbers as label of texts
        self.tokenizer = tokenizer #tokenizer for converting text to tokens
        self.max_len = max_len #the set maximum inout sequence length of tokens 

    def __len__(self):
        return len(self.texts) #this method returns number of samples in the dataset aka length of the dataset

    def __getitem__(self, idx): 
        #this method retrieves a specific text and its corresponding label, tokenizes the text 
        # and makes the sequence lengths consistent through padding and truncation for model input
        
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
            'attention_mask': encoding['attention_mask'].squeeze(0),  
            #returns attention mask as a tensor, attention masks indicates which 
            # tokens in the input sequence are actual tokens (1) and which are padding tokens (0)
            
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

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed) 
#instantiates stratified K-fold cross-validation from StratifiedKFold library
#shuffles the data to ensure randomness and avoid bias due to the order of the dataset
#random_state=seed is the fixed seed for the random number generator, ensuring that the splitting of dataset are the same across multiple runs

fold_results = [] #empty list to store performance metrics at the result for each fold
aggregate_cm = np.zeros((len(label_map), len(label_map)))  #initialize a zero matrix for the aggregated confusion matrices across folds

# === Hyperparameters ===
batch_size = 16
#number of abstract text samples in a batch during training and validation, a batch is a subset of 
# the dataset used to train the model in one forward and backward pass, the model processes this amount of text abstracts at a time during training or evaluation

epochs = 50  #maximum number of epochs to train the model, if early stopper is implemented it will rarely reach a high number of epoch
max_len = 512  #maximum sequence length for tokenization
early_stopping_patience = 10  #number of epochs to wait for improvement before stopping training early
scheduler_patience = 2  #number of epochs to wait before reducing the learning rate when loss value stagnates
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #makes sure GPU is used if available for faster training and uses CPU otherwise


# === Start K-Fold Cross-Validation ===
with open("metrics-earlystopping=10-lrscheduled-BERT-uncased-batch16.txt", "w") as log_file:  #opens log file in writing mode to the variable log_file for writing performance metrics
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, numeric_labels)):
        #loops over each fold generated by skf.split() to perform cross-validation the dataset is split into training (train_idx) 
        # and validation (val_idx) subsets which it then uses to train and evaluate/validate the model
        
        #skf.split() uses the StratifiedKFold instance (skf) to split the dataset into training and validation indices for each fold
        #train_index and val_index are separate tuples containing lists of indices of samples used for training (train_index) and validating (val_index) in this fold
        #each fold ensures that there are no overlap between the training and validation indices, ensuring the model is trained on one part of the dataset and validated on another without overlap
        
        print(f"Fold {fold + 1}/{k_folds}")  #prints the current fold in the terminal during training
        log_file.write(f"Fold {fold + 1}/{k_folds}\n")  #logs the current fold number to log_file

        #split data into training and validation sets using indices from skf.split()
        train_texts = [texts[i] for i in train_idx]  #extracts training texts based on indices
        train_labels = [numeric_labels[i] for i in train_idx]  #extracts the corresponding training labels
        val_texts = [texts[i] for i in val_idx]  #extracts validation texts based on indices
        val_labels = [numeric_labels[i] for i in val_idx]  #extracts the corresponding validation labels.
        
        #creates PyTorch datasets by instantiating AbstractDataset class
        train_dataset = AbstractDataset(train_texts, train_labels, tokenizer, max_len)  #this dataset is for training, it stores train_texts and train_labels, tokenizes it through tokenizer, and returns the tokenized text and labels
        val_dataset = AbstractDataset(val_texts, val_labels, tokenizer, max_len)  #this dataset is for validation, it stores val_texts and val_labels, tokenizes it through tokenizer, and returns the tokenized text and labels

        #creates DataLoader objects for operations such as batching and shuffling, making it ready for model input. 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  #feeds batches of training data to the model enabling gradient updates each batch processes
        #the shuffling here ensures that the model sees the training data in a different order in each epoch to reduce likelihood of overfitting of a specific sequence
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size)  #feeds batches of validation data to evaluate model performance after each epoch, this doesnt shuffle as the order does not affect performance evaluation

        # === Loading BERT Model ===
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))  #loads Bert-base uncased version as the language model trained here. it is uncased (ignores cases) matching the tokenizer.
        model.to(device)  #moves the model to be computed on the GPU or CPU as available

        optimizer = AdamW(model.parameters(), lr=2e-5)  
        #instantiates Adam optimizer for updating the parameters (weights and biases) during traning, extra detail of adamw algorithm is in the import section
        #model.parameters() refers to all trainable parameters of the model, updatable by the optimizer
        #the optimizer is initialized with a learning rate of 2e-5
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=0.1)  
        #initializes learning rate scheduler "ReduceLROnPlateau", its job is to adjusts learning rate during training to improve model convergence
        #mode='min': min mode, scheduler reduces the learning rate if the monitored metric (metric being the loss, which are expected to when over training) stops decreasing. so if the loss stops decreasing, it triggers the scheduler to reduce the learning rate.
        #patience=scheduler_patience sets the number of epochs to wait after the metric stops improving before reducing the learning rate (according to scheduler_patience value set in hyperparameters)
        #factor=0.1 is the factor by which the learning rate is multiplied when it is reduce

        # Early Stopping Variables
        best_val_loss = float('inf')  
        # Initialize the best validation loss to infinity which will be replaced by a smaller (therefore better) loss that is calculated at the end of each epoch
        # The function is track the smallest validation loss seen so far, helps in early stopping 
        # Early stopping uses validation loss instead of training loss because it evaluates the model's ability to generalize on unseen data, while training loss only measures performance on the training set
        patience_counter = 0  # a simple counter, will keep track of the number of consecutive epochs where the validation loss does not improve

        # Training Loop
        for epoch in range(epochs): #Loops 'epochs' amount of times
            print(f"Epoch {epoch + 1}/{epochs}")  #prints the current epoch
            log_file.write(f"  Epoch {epoch + 1}/{epochs}\n")  #write the current epoch to log_file

            # === Training phase ===
            model.train()  # Set the model to training mode, this is a PyTorch library method. It enables layers like dropout and batch normalization to function
            epoch_loss = 0  # Initialize cumulative loss for the epoch.
            for batch in train_loader: #iterates thorugh training data in batches
                optimizer.zero_grad()  #resets the gradients (clear old gradients) of all model parameters before performing a new backpropagation step, ensuring correct updates for each training step

                input_ids = batch['input_ids'].to(device)  #transfers the tokenized input sequences (input_ids) to the device (GPU) for model computation.
                attention_mask = batch['attention_mask'].to(device)  #moves the attention mask to the same device as the model
                labels = batch['labels'].to(device)  #moves labels to the device
                #these ensure all batch components (inputs and labels) are moved to the same device as the model

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  
                #perform forward pass through BERTForSequenceClassification model, the input (input_ids and attention_mask) and labels are fed into the model to compute predictions and loss
                #BERT encoder processes the input_ids and attention_mask to generate contextualized embeddings for the input tokens
                #returns an outputs object containing loss and logits
            
                loss = outputs.loss  #extracts the computed loss from outputs object, loss is calculated using the model's predictions and the target label
                epoch_loss += loss.item()  #converts the loss from a tensor to a Python scalar using .item() and accumulates it for logging
                loss.backward()  #performs backward pass and backpropagation, calculating the gradients of the loss with respect to the model's parameters
                optimizer.step()  #updates model parameters based on the gradient computed in backpropagation with optimizer

            average_epoch_loss = epoch_loss / len(train_loader)  #calculate average training loss for the epoch by dividing accumulated loss (epoch_loss) by the number of batches in train_loader
            print(f"Training Loss after epoch {epoch + 1}: {average_epoch_loss}")  #print training loss at current epoch
            log_file.write(f"    Training Loss: {average_epoch_loss:.4f}\n")  #writes average training loss as 4 decimal points floats to log_file 

            # === Validation Phase ===
            model.eval()  #Set the model to evaluation mode. This is also a PyTorch library method. Here, layers like dropout and batch normalization switch to fixed behavior (no dropout and fixed statistics)
            val_loss = 0  # Initialize a variable to accumulate the validation loss for the current epoch
            
            val_preds = []  #creates an empty list to store the model’s predictions for the validation data
            val_true = []  #creates an empty list to store true validation labels
            #values in these two lists will be compared for validation metrics (F1, accuracy, precision)
            
            with torch.no_grad():  #disables gradient calculation since validation phase is a forward pass only process testing performance on unseen data, which doesnt update any parameters and there fore doesnt use gradients
                for batch in val_loader: #iterates through the validation data in batches
                    
                    input_ids = batch['input_ids'].to(device)  #move input IDs to the device (GPU)
                    attention_mask = batch['attention_mask'].to(device)  #move attention mask to the device
                    labels = batch['labels'].to(device)  #moves labels to the device
                    #same process like the one in training phase, moving all components to device for operations.

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  
                    #perform forward pass through BERTForSequenceClassification model, the input (input_ids and attention_mask) and labels are fed into the model to compute predictions and loss
                    #BERT encoder processes the input_ids and attention_mask to generate contextualized embeddings for the input tokens
                    #returns an outputs object containing loss and logits
                    
                    loss = outputs.loss  #extracts the computed loss from outputs object, loss is calculated using the model's predictions and the target label
                    val_loss += loss.item()  #converts the loss from a tensor to a Python scalar using .item() and accumulates it for logging

                    logits = outputs.logits  #extracts logits which contains the raw predictions values from the model's classification head output to logits tensor
                    preds = torch.argmax(logits, dim=-1)  
                    # .argmax finds the index of the maximum value in logits, specifying the location as the last dimension of the tensor
                    # why last dimension? logits are structured as a tensor of shape (batch_size, num_classes) and each row corresponds to a sample, and each column corresponds to the raw score for a specific class. 
                    # The last dimension (dim=-1) contains the class scores for each sample
                    # .argmax finds largest value class scores for each sample as the prediction of the model
                    
                    val_preds.extend(preds.cpu().numpy()) # appends the predictions for the current batch to the val_preds list
                    val_true.extend(labels.cpu().numpy())  # stores the true class labels in val_true list
                    # .cpu() is used to move them to CPU device, because .numpy() calculation can only be done by the CPU
                    # then .numpy() itself converts the tensor into a NumPy array.
                    # these lists will later be used to calculate validation metrics

            average_val_loss = val_loss / len(val_loader)  #computes the average validation loss by dividing the total validation loss by the number of batches in the validation set
            scheduler.step(average_val_loss)  
            #This adjusts the learning rate using the ReduceLROnPlateau learning rate scheduler based 
            # on the average validation loss when the validation loss fails to improve after a certain number of epochs set by "patience"
            print(scheduler.get_last_lr()) # prints most recent learning rate used by the optimizer

            print(f"Validation Loss after epoch {epoch + 1}: {average_val_loss}")  # Prints validation loss at currecnt epoch
            log_file.write(f"    Validation Loss: {average_val_loss:.4f}\n")  # Log validation loss of current epoch to log_file as 4 decimal floats

            # === Early Stopping Check ===
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss  # update the best validation loss if average_val_loss of current epoch is smaller then best_val_loss
                patience_counter = 0  # reset the patience counter
                #torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')  #save the model checkpoint for this fold to keep trach of the best performing model #currently commented out for storage concern since we have a final saving anyway
            else:
                patience_counter += 1  
                # if average_val_loss is not smaller than best_val_loss then increment patience_counter, this allows the model to continue training 
                #for a few more epochs even if it doesn't improve until it maxes the patience_counter
                
                print(f"Early stopping patience counter: {patience_counter}/{early_stopping_patience}")  # print the current patience counter value
                if patience_counter >= early_stopping_patience:  # if the counter reached maximum early stopping patience, it then triggers early stopping and stops training in new epochs
                    print("Early stopping triggered.") #Informs in terminal that early stopping is triggered.
                    break

        # === Calculate Metrics ===
        # calculation are done based on the two lists (val_true, val_preds) using methods from sklearn.metrics library
        accuracy = accuracy_score(val_true, val_preds)  #compute accuracy for this fold 
        precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='weighted')  #compute precision, recall,  F1 score
        # the _ is used to as an ignoring placeholder of the fourth return values "Support" of precision_recall_fscore_support method. 
        # Support is the number of true instances for each label, it is unused as a metrics here
        
        cm = confusion_matrix(val_true, val_preds)  #compute confusion matrix

        print(f"Fold {fold + 1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")  # Print calculated metrics.
        log_file.write(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")  # Log calculated metrics.
        aggregate_cm += cm  # current confusion matrix values are added to aggregate_cm, the combined confusion matrix across all folds.

        fold_results.append((accuracy, precision, recall, f1))  #Adds a tuple containing the performance metrics of the current fold to fold_results list.

    # === Aggregate Confusion Matrix ===
    classes = list(label_map.keys())  #retrieve class names from the label mapping in the key portion of label_map dictionary
    plot_confusion_matrix(aggregate_cm, classes, "Aggregate Confusion Matrix (Percentage)", "aggregate-confusion-metrics-earlystopping=10-lrscheduled-SciBERT-uncased-batch16.jpeg")  
    #calls the custom plot_confusion_matrix function to plot and save the confusion matrix.

    # === Calculate and Save Average Metrics ===
    log_file.write("\nAverage Metrics Across All Folds:\n")  # Writes a header for average metrics.
    
    if fold_results: #checks if fold_results contain any values, ensures that it only averages if there are metrics results
        avg_accuracy = np.mean([result[0] for result in fold_results])  # Compute average accuracy with numpy
        avg_precision = np.mean([result[1] for result in fold_results])  # Compute average precision with numpy
        avg_recall = np.mean([result[2] for result in fold_results])  # Compute average recall with numpy
        avg_f1 = np.mean([result[3] for result in fold_results])  # Compute average F1 score with numpy
    else:
        avg_accuracy = avg_precision = avg_recall = avg_f1 = 0.0  # Sets the values to 0 if no metrics are available in fold_results

    log_file.write(f"  Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}\n")  # writes the average metrics in log_file

model.save_pretrained('./fine_tuned_bert')  # Save the fine-tuned BERT model
tokenizer.save_pretrained('./fine_tuned_bert')  # Save the tokenizer used during training to ensure tokenizer and tokenization consistency