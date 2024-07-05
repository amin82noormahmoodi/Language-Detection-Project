Sure, here's the README file for your project in English:

---

# Language Identification (LID) using RNN and Transformer Models

This repository contains two approaches for Language Identification (LID) tasks:
1. **RNN-based model using GRU**
2. **Transformer-based model using BERT**

## Table of Contents
1. [Dataset](#dataset)
2. [RNN-based Model](#rnn-based-model)
3. [Transformer-based Model](#transformer-based-model)
4. [Training and Evaluation](#training-and-evaluation)
5. [Results](#results)
6. [Requirements](#requirements)
7. [Usage](#usage)

## Dataset

We use two datasets:
1. `Document_level_data.csv` for the RNN model.
2. `Span_level_data.csv` for the Transformer model.

Both datasets include text samples and their corresponding language labels.

## RNN-based Model

The RNN model is implemented using Gated Recurrent Units (GRU). Below is a summary of the model and the data preprocessing steps:

### Preprocessing

1. **Dataset Initialization**:
   - The dataset is read from a CSV file and split into training, validation, and test sets.
   - The text data is tokenized into n-grams for model input.

2. **Vocabulary Creation**:
   - N-grams are generated from the text data.
   - A vocabulary is built from these n-grams.

3. **Data Loading**:
   - A custom `collate_batch` function is used to prepare batches of data for training and evaluation.
   - DataLoader instances are created for training, validation, and test sets.

### Model Architecture

- Embedding layer
- GRU layer (bidirectional)
- Linear layer for classification

### Training and Evaluation

The model is trained using the AdamW optimizer and CrossEntropyLoss. The training and evaluation functions handle the forward pass, loss computation, backpropagation, and accuracy calculation.

## Transformer-based Model

The Transformer model uses BERT (bert-base-multilingual-cased) for token classification.

### Preprocessing

1. **Dataset Initialization**:
   - The dataset is read from a CSV file and split into training and test sets.
   - The text data and labels are tokenized using the BERT tokenizer.

2. **Label Alignment**:
   - Labels are aligned with the tokenized inputs to ensure that only the first token of each word is labeled.

### Model Architecture

- BERT for token classification with 10 output labels (languages).

### Training and Evaluation

The model is trained using the `Trainer` class from the Transformers library. Training arguments are set for learning rate, batch size, number of epochs, and other hyperparameters. Evaluation metrics include precision, recall, F1 score, and accuracy.

## Results

### RNN-based Model

The performance of the RNN model is evaluated in terms of training loss, validation loss, and accuracy. The model achieves high accuracy on the validation set, indicating its effectiveness for the LID task.

### Transformer-based Model

The Transformer model is evaluated using seqeval for token classification. The model achieves the following results:

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1    | Accuracy |
|-------|----------------|-----------------|-----------|--------|-------|----------|
| 1     | 0.007400       | 0.007768        | 0.990285  | 0.991569 | 0.990927 | 0.998471 |
| 2     | 0.008100       | 0.003942        | 0.995789  | 0.996757 | 0.996273 | 0.999197 |

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- scikit-learn
- pandas
- numpy
- tensorboard
- evaluate

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### RNN-based Model

1. **Prepare the dataset**: Place `Document_level_data.csv` in the root directory.
2. **Run the training script**:
    ```bash
    python train_rnn.py
    ```

### Transformer-based Model

1. **Prepare the dataset**: Place `Span_level_data.csv` in the root directory.
2. **Run the training script**:
    ```bash
    python train_transformer.py
    ```

Training logs and results will be available in the console and TensorBoard.

---

This README provides an overview of the project, including the dataset, model architectures, training procedures, and results. Follow the instructions to reproduce the experiments and achieve similar results.
