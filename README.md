README:

---

# Language Identification (LID) using RNN and Transformer Models

This repository contains three approaches for Language Identification (LID) tasks:
1. **RNN-based model using GRU**
2. **Transformer-based model using BERT**
3. **Overall Token Classification Framework**

## Table of Contents
1. [Dataset](#dataset)
2. [RNN-based Model](#rnn-based-model)
   - [Preprocessing](#preprocessing)
   - [Model Architecture](#model-architecture)
   - [Mathematical Explanation](#mathematical-explanation)
3. [Transformer-based Model](#transformer-based-model)
   - [Preprocessing](#preprocessing-1)
   - [Model Architecture](#model-architecture-1)
   - [Mathematical Explanation](#mathematical-explanation-1)
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

The RNN model is implemented using Gated Recurrent Units (GRU). Below is a summary of the model, the data preprocessing steps, and the mathematical explanation.

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

- **Embedding layer**: Converts n-gram tokens into dense vectors.
- **GRU layer (bidirectional)**: Processes the sequence of embeddings to capture contextual information.
- **Linear layer for classification**: Outputs the language prediction probabilities.

### Mathematical Explanation

#### Embedding Layer

Given an input sequence of tokens `{x_1, x_2, \ldots, x_T}`, the embedding layer maps each token `x_t` to a dense vector `e_t` in `R^d`.

```
e_t = Embedding(x_t)
```

#### GRU Layer

The GRU layer processes the sequence of embeddings `{e_1, e_2, \ldots, e_T}` and computes hidden states `{h_1, h_2, \ldots, h_T}`. For a bidirectional GRU, we have forward and backward passes:

```
h_t^{fwd} = GRU_{fwd}(e_t, h_{t-1}^{fwd})
h_t^{bwd} = GRU_{bwd}(e_t, h_{t+1}^{bwd})
```

The final hidden state is a concatenation of forward and backward hidden states:

```
h_t = [h_t^{fwd}; h_t^{bwd}]
```

#### Linear Layer for Classification

The concatenated hidden state `h_t` is passed through a linear layer followed by a softmax activation to obtain the probability distribution over language labels:

```
y_t = Softmax(W h_t + b)
```

where `W` and `b` are learnable parameters.

## Transformer-based Model

The Transformer model uses BERT (bert-base-multilingual-cased) for token classification. Below is a summary of the model, the data preprocessing steps, and the mathematical explanation.

### Preprocessing

1. **Dataset Initialization**:
   - The dataset is read from a CSV file and split into training and test sets.
   - The text data and labels are tokenized using the BERT tokenizer.

2. **Label Alignment**:
   - Labels are aligned with the tokenized inputs to ensure that only the first token of each word is labeled.

### Model Architecture

- **BERT for token classification**: The BERT model processes the input tokens and outputs contextualized embeddings.
- **Classification Layer**: Outputs the language prediction probabilities for each token.

### Mathematical Explanation

#### Tokenization and Embeddings

Given an input sequence of tokens `{x_1, x_2, \ldots, x_T}`, the BERT tokenizer maps each token to an embedding `e_t` in `R^d`:

```
e_t = BERT_Embedding(x_t)
```

#### Transformer Encoder

The Transformer encoder processes the sequence of embeddings `{e_1, e_2, \ldots, e_T}` using self-attention and feedforward layers to compute contextualized embeddings `{h_1, h_2, \ldots, h_T}`.

For each layer in the Transformer:

1. **Self-Attention**:
   
   The self-attention mechanism computes a weighted sum of the input embeddings, where the weights are derived from the input itself:

   ```
   Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k)) V
   ```

   where `Q`, `K`, `V` are the query, key, and value matrices derived from the input embeddings.

2. **Feedforward Network**:
   
   The feedforward network applies two linear transformations with a ReLU activation in between:

   ```
   FFN(h) = ReLU(W_1 h + b_1) W_2 + b_2
   ```

#### Classification Layer

The final contextualized embeddings `h_t` are passed through a linear layer followed by a softmax activation to obtain the probability distribution over language labels for each token:

```
y_t = Softmax(W h_t + b)
```

## Training and Evaluation

Both models are trained using the AdamW optimizer and CrossEntropyLoss. The training and evaluation functions handle the forward pass, loss computation, backpropagation, and accuracy calculation.

### Evaluation Metrics

The evaluation metrics include precision, recall, F1 score, and accuracy. These metrics are computed using the seqeval library for the token classification task.

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
