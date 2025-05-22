# DA6401-Assignment-3
This project implements a sequence-to-sequence (Seq2Seq) RNN model for character-level transliteration using the Dakshina dataset. The system can transliterate text from Latin script to Malayalam script.

## Project Structure
```
├── predictions_attention/
├── predictions_vanilla/
├── dataset.py                  # Dataset handling and preprocessing
├── vanilla_model.py            # Vanilla Seq2Seq model implementation (Encoder-Decoder)
├── training.py                 # Training, evaluation, and inference functions for vanilla model
├── attention_model.py          # Attention-based Seq2Seq model implementation
├── training_attention.py       # Training, evaluation, and inference functions for attention model
├── visualize_attention.py      # Attention visualization utilities
├── connectivity.py             # Connectivity visualization utilities
├── README.md                   # This file
├── vanilla.ipynb               # Main training and experimentation notebook for vanilla models
├── attention.ipynb             # Main training and experimentation notebook for attention models
├── best_vanilla_model.pth      # Best vanilla model weights obtained from sweep
├── best_attention_model.pth    # Best attention model weights obtained from sweep
└── .gitignore
```

## Requirements
Run `pip install -r requirements.txt to install all the required libraries to run this project`

### Data Structure:
The model expects data in the following format:
```
अजनबी    ajanabee
घर  ghar
```
Column 0: Target script (e.g., Devanagari, Malayalam)
Column 1: Source script (Latin/Roman)
Both columns are seperated by a tab

## Function Files
### 1. Dataset (dataset.py)
1. `TransliterationDataset` Class:
    - Purpose: Handles data loading and preprocessing for character-level transliteration
    - Functionality: Creates character-level vocabularies for both source and target languages
    - Special Tokens: Implements <PAD>, <SOS>, <EOS>, <UNK> tokens with fixed indices
    - Preprocessing: Strips whitespace, applies lowercase conversion, handles missing data
    - Vocabulary Building: Creates bidirectional character-to-index mappings for both languages
    - Sequence Processing: Adds start/end tokens, handles sequence truncation based on max_len
    - Encoding/Decoding: Provides methods to convert between text and numerical representations
    - Key Methods:
        - `__init__()`: Initializes dataset, builds vocabularies, processes data file
        - `__len__()`: Returns dataset size
        - `__getitem__()`: Returns processed sample with source/target tensors and metadata
        - `get_vocab_size()`: Returns vocabulary size for source or target language
        - `decode_indices()`: Converts numerical indices back to readable text

2. `collate_fn` Function:
    - Purpose: Custom batch processing for DataLoader
    - Sorting: Sorts batch by sequence length for efficient packed sequence processing
    - Padding: Pads sequences to uniform length within batch using PAD tokens
    - **Output:** Returns batched tensors with source, target, lengths, and original texts

### 2. Vanilla Model Architecture (vanilla_model.py)
1. `Encoder` Class:
    - Purpose: Processes input sequences to create contextualized representations
    - Cell Support: Supports RNN, LSTM, and GRU cells with configurable parameters
    - Architecture: Embedding layer → RNN layers → Output hidden states
    - Efficiency: Uses packed padded sequences to handle variable-length inputs
    - **Output:** Returns all hidden states and final hidden state for decoder initialization

2. `Decoder` Class:
    - Purpose: Generates output sequences one token at a time
    - Architecture: Embedding layer → RNN layers → Linear output layer
    - Generation: Takes previous token and hidden state, outputs next token probability
    - Flexibility: Configurable cell type, layers, dropout, and hidden dimensions

3. `Seq2Seq` Class:
    - Purpose: Combines encoder and decoder into complete sequence-to-sequence model
    - Training: Implements teacher forcing with configurable probability
    - Forward Pass: Encodes source sequence, then decodes target sequence step by step
    - Inference: Supports beam search and greedy decoding for prediction generation
    - Compatibility: Ensures encoder and decoder use same cell type and dimensions

### 3. Attention Model Architecture (attention_model.py)
1. `Attention` Class:
    - Purpose: Implements Bahdanau (additive) attention mechanism
    - Computation: Calculates attention weights between decoder state and all encoder outputs
    - Architecture: Linear transformations → tanh activation → softmax normalization
    - **Output:** Returns attention weights indicating which input positions to focus on

2. `Encoder` Class:
    - Purpose: Same as vanilla encoder - processes input sequences
    - Reusability: Identical implementation to vanilla model for consistency
    - **Output:** Returns all encoder hidden states needed for attention computation

3. `AttentionDecoder` Class:
    - Purpose: Enhanced decoder that incorporates attention mechanism
    - Input: Takes embedding, previous hidden state, and all encoder outputs
    - Attention Integration: Computes attention weights and context vector at each step
    - Architecture: More complex than vanilla decoder due to attention components
    - **Output:** Returns predictions, updated hidden state, and attention weights for visualization

4. `AttentionSeq2Seq` Class:
    - Purpose: Complete attention-based sequence-to-sequence model
    - Enhancement: Extends vanilla seq2seq with attention mechanism
    - Training: Maintains teacher forcing while computing attention at each step
    - Inference: Generates predictions while tracking attention weights for analysis
    - Visualization: Returns attention weights history for creating heatmaps

### 4. Training Functions (training.py & training_attention.py)
1. `train()` Function:
    - Purpose: Handles model training for one epoch
    - Process: Forward pass → loss calculation → backpropagation → parameter updates
    - Teacher Forcing: Uses ground truth tokens with specified probability during training
    - Optimization: Implements gradient clipping to prevent exploding gradients
    - Loss Handling: Ignores padding tokens in loss calculation, skips SOS token in targets

2. `evaluate()` Function:
    - Purpose: Evaluates model performance without gradient updates
    - Mode: Sets model to evaluation mode, disables teacher forcing
    - Metrics: Calculates average loss across validation/test dataset
    - Memory: Uses torch.no_grad() context for memory efficiency

3. `transliterate()` Function:
    - Purpose: Performs inference on individual source texts
    - Process: Converts text to indices → encodes → decodes with greedy search
    - Stopping: Continues until EOS token or maximum length reached
    - **Output:** Returns transliterated text by converting indices back to characters

4. `calculate_accuracy()` Function:
    - Purpose: Computes exact match accuracy across entire dataset
    - Evaluation: Processes all test samples and compares predictions to ground truth
    - Metrics: Returns accuracy percentage and list of all predictions for analysis
    - Storage: Saves predictions with source, predicted, and target texts for review

### 5. Attention Visualization (visualize_attention.py)
1. `generate_attention_heatmaps()` Function:
    - Purpose: Creates grid visualization of attention patterns for multiple samples
    - Layout: Arranges multiple attention heatmaps in organized grid format
    - Visualization: Uses color intensity to show attention strength between source and target characters
    - Font Support: Handles complex scripts with appropriate font configuration
    - **Output:** Saves comprehensive grid visualization for multiple sample analysis

2. `create_individual_attention_plots()` Function:
    - Purpose: Generates detailed individual attention visualizations
    - Detail Level: Provides more detailed view compared to grid visualization
    - Annotations: Includes numerical attention values within heatmap cells
    - Comparison: Shows source text, prediction, target text, and correctness information
    - Quality: Higher resolution individual plots for detailed analysis

3. `transliterate_with_attention()` Function:
    - Purpose: Performs transliteration while capturing attention weights
    - **Dual Output:** Returns both transliterated text and attention weight matrices
    - Visualization Ready: Formats attention weights for direct use in plotting functions
    - Integration: Specifically designed to work with attention model's inference method

4. `plot_attention()` Function:
    - Purpose: Core plotting utility for creating attention heatmaps
    - Customization: Configurable titles, labels, and color schemes
    - Character Mapping: Properly aligns source and target characters in visualization
    - Export: Returns matplotlib figure object for further customization or saving

### 6. Connectivity Visualization (connectivity.py)
1. `generate_connectivity_visualization()` Function:
    - Purpose: Creates interactive HTML visualization showing attention connectivity between input and output characters
    - Interactivity: Hover over output characters to see which input characters receive attention
    - Visual Design: Modern, responsive design with gradient backgrounds and smooth animations
    - Data Processing: Handles model predictions, attention weight extraction, and normalization
    - **Output:** Returns complete HTML string with embedded CSS and JavaScript

2. `save_connectivity_visualization()` Function:
    - Purpose: Generates and saves the interactive connectivity visualization as an HTML file
    - File Management: Handles file writing with proper encoding for complex scripts
    - Integration: Works with trained attention models and dataset instances
    - Convenience: Wrapper function that combines generation and saving in one call

3. `batch_connectivity_visualization()` Function:
    - Purpose: Generates connectivity visualization from a DataLoader with multiple samples
    - Batch Processing: Efficiently processes multiple samples from test dataset
    - Sample Selection: Configurable number of samples to include in visualization
    - Data Extraction: Extracts source texts from batched data with proper decodings

4. `generate_heatmap_html()` Function:
    - Purpose: Creates HTML representation of attention weights as a color-coded heatmap
    - Grid Layout: Uses CSS Grid for responsive and aligned heatmap display
    - Color Coding: Maps attention weights to color intensity for intuitive visualization
    - Tooltips: Provides detailed attention values on hover for precise analysis

## Model Training
Follow the notebooks vanilla.ipynb and attention.ipynb to train and evaluate the vanilla model and attention model respectively.