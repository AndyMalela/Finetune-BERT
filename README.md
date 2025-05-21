# Fine-Tuning BERT for Abstract Classification

This project fine-tunes a pre-trained BERT model to perform the classic text classification task. 
In this case it classifies academic abstracts into three disciplines: **Psychology**, **Political Science**, and **Sociology**. Dataset contains labeled abstracts scrapped from google scholar.

### Notes

- This was a course project for experimenting with modern NLP model using transformers.

├── FinetuningBERT.py # Main training script
├── psycho-polsci-socio.csv # Input data file
├── metrics-earlystopping=10-.txt # Logged training results
├── fine_tuned_bert/ # Saved model and tokenizer
├── aggregate-confusion-.jpeg # Confusion matrix visualization
└── README.md # This file

Model used is `bert-base-uncased` using Hugging Face's Transformers and PyTorch.

Implements:
- Stratified K-Fold Cross-Validation
- Early Stopping with patience
- Learning Rate Scheduler (`ReduceLROnPlateau`)



### 🚀 How to Run

```bash
pip install transformers torch scikit-learn pandas seaborn matplotlib
python FinetuningBERT.py
