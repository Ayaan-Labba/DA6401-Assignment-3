# DA6401-Assignment-3
This project implements a sequence-to-sequence (Seq2Seq) RNN model for character-level transliteration using the Dakshina dataset. The system can transliterate text from Latin script to Malayalam script.

## Project Structure
├── dataset.py          # Dataset handling and preprocessing
├── vanilla_model.py    # Seq2Seq model implementation (Encoder-Decoder)
├── training.py         # Training, evaluation, and inference functions
├── README.md          # This file
└── [Jupyter Notebook] # Main training and experimentation notebook

## Requirements
Run `pip install -r requirements.txt to install all the required libraries to run this project`


Data Structure:
The model expects data in the following format:
target_script_word    latin_script_word
अजनबी                ajanabee
घर                   ghar
Column 0: Target script (e.g., Devanagari, Malayalam)
Column 1: Source script (Latin/Roman)

## Base Files
1. Dataset (dataset.py)
`TransliterationDataset` Class:
- Handles data loading and preprocessing
- Creates character-level vocabularies for source and target languages
- Implements special tokens: <PAD>, <SOS>, <EOS>, <UNK>
- Provides encoding/decoding functionality

Key Parameters:
- data_path: Path to the dataset file
- max_len: Maximum sequence length (default: 50)
- lowercase: Whether to lowercase Latin text (default: True)

2. Model Architecture (vanilla_model.py)
Components:
- Encoder: RNN/LSTM/GRU that processes input sequences
- Decoder: RNN/LSTM/GRU that generates output sequences
- Seq2Seq: Combined encoder-decoder architecture

Supported Cell Types:
Vanilla RNN ('rnn')
LSTM ('lstm')
GRU ('gru')

3. Training & Evaluation (training.py)
Functions:
- train(): Training loop with teacher forcing
- evaluate(): Evaluation loop
- transliterate(): Single sequence inference
- calculate_accuracy(): Batch accuracy calculation

