import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        """
        Args:
            encoder_hidden_size: Size of the encoder's hidden state
            decoder_hidden_size: Size of the decoder's hidden state
        """
        super(Attention, self).__init__()
        
        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: Decoder hidden state of shape [batch_size, decoder_hidden_size]
            encoder_outputs: Encoder outputs of shape [batch_size, src_len, encoder_hidden_size]
        
        Returns:
            attention_weights: Attention weights of shape [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Create energy tensor
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Calculate attention weights
        attention = self.v(energy).squeeze(2)
        
        # Return softmaxed attention weights
        return F.softmax(attention, dim=1)

# Encoder class (same as before)
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout=0, cell_type='lstm'):
        """
        Args:
            input_size: Size of the vocabulary
            embedding_size: Size of the embeddings
            hidden_size: Size of the hidden state
            n_layers: Number of layers in the RNN
            dropout: Dropout probability
            cell_type: Type of RNN cell ('rnn', 'lstm', 'gru')
        """
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_size, 
                hidden_size, 
                num_layers=n_layers, 
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                embedding_size, 
                hidden_size, 
                num_layers=n_layers, 
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        else:  # default to vanilla RNN
            self.rnn = nn.RNN(
                embedding_size, 
                hidden_size, 
                num_layers=n_layers, 
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
    
    def forward(self, x, lengths):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len]
            lengths: Length of each sequence in the batch
        
        Returns:
            outputs: Tensor of shape [batch_size, seq_len, hidden_size]
            hidden: Last hidden state
        """
        # Embedding
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, embedding_size]
        
        # Pack padded sequences for efficiency
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # Pass through RNN
        if self.cell_type == 'lstm':
            outputs, (hidden, cell) = self.rnn(packed)
            # Unpack the sequence
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.rnn(packed)
            # Unpack the sequence
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            return outputs, hidden

# Attention-based Decoder
class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, encoder_hidden_size, decoder_hidden_size, n_layers=1, dropout=0, cell_type='lstm'):
        """
        Args:
            output_size: Size of the vocabulary
            embedding_size: Size of the embeddings
            encoder_hidden_size: Size of the encoder's hidden state
            decoder_hidden_size: Size of the decoder's hidden state
            n_layers: Number of layers in the RNN
            dropout: Dropout probability
            cell_type: Type of RNN cell ('rnn', 'lstm', 'gru')
        """
        super(AttentionDecoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # Attention mechanism
        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)
        
        # RNN layer - takes embedding + weighted sum of encoder outputs
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_size + encoder_hidden_size, 
                decoder_hidden_size, 
                num_layers=n_layers, 
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                embedding_size + encoder_hidden_size, 
                decoder_hidden_size, 
                num_layers=n_layers, 
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        else:  # default to vanilla RNN
            self.rnn = nn.RNN(
                embedding_size + encoder_hidden_size, 
                decoder_hidden_size, 
                num_layers=n_layers, 
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        
        # Output layer
        self.fc_out = nn.Linear(decoder_hidden_size + embedding_size + encoder_hidden_size, output_size)
    
    def forward(self, x, hidden, encoder_outputs):
        """
        Args:
            x: Tensor of shape [batch_size, 1]
            hidden: Hidden state from the previous time step
            encoder_outputs: Outputs from the encoder
        
        Returns:
            output: Tensor of shape [batch_size, output_size]
            hidden: Updated hidden state
            attention_weights: Attention weights for visualization
        """
        # Embedding
        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embedding_size]
        
        # Calculate attention weights
        if self.cell_type == 'lstm':
            attention_weights = self.attention(hidden[0][-1], encoder_outputs)
        else:
            attention_weights = self.attention(hidden[-1], encoder_outputs)
        
        # Reshape attention weights to use for bmm
        attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        
        # Calculate weighted sum of encoder outputs (context vector)
        context = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, encoder_hidden_size]
        
        # Concatenate embedding and context vector
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, embedding_size + encoder_hidden_size]
        
        # Pass through RNN
        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(rnn_input, hidden)
            # Concatenate RNN output, embedding and context for output layer
            output = torch.cat((output, embedded, context), dim=2)
            # Convert output to vocabulary distribution
            prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
            return prediction, (hidden, cell), attention_weights.squeeze(1)
        else:
            output, hidden = self.rnn(rnn_input, hidden)
            # Concatenate RNN output, embedding and context for output layer
            output = torch.cat((output, embedded, context), dim=2)
            # Convert output to vocabulary distribution
            prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
            return prediction, hidden, attention_weights.squeeze(1)

# Seq2Seq model with attention
class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(AttentionSeq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Ensure the encoder and decoder have compatible hidden dimensions
        assert encoder.hidden_size == decoder.encoder_hidden_size, \
            "Encoder hidden size and decoder encoder_hidden_size must be equal!"
        
        # Ensure they use the same cell type
        assert encoder.cell_type == decoder.cell_type, \
            "Encoder and decoder must use the same RNN cell type!"
    
    def forward(self, source, source_length, target, teacher_forcing_ratio=0.5):
        """
        Args:
            source: Tensor of shape [batch_size, source_len]
            source_length: Length of each source sequence
            target: Tensor of shape [batch_size, target_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Tensor of shape [batch_size, target_len, output_size]
            attention_weights_history: Attention weights for all time steps
        """
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        max_src_len = source.shape[1]
        
        # Tensor to store decoder outputs and attention weights
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        attention_weights_history = torch.zeros(batch_size, target_len, max_src_len).to(self.device)
        
        # Encode the source sequence
        if self.encoder.cell_type == 'lstm':
            encoder_outputs, (hidden, cell) = self.encoder(source, source_length)
            decoder_hidden = (hidden, cell)
        else:
            encoder_outputs, hidden = self.encoder(source, source_length)
            decoder_hidden = hidden
        
        # First input to the decoder is the < SOS > token
        decoder_input = target[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Teacher forcing: use the ground-truth target as the next input
        # with probability teacher_forcing_ratio
        for t in range(1, target_len):
            # Pass through decoder with attention
            if self.decoder.cell_type == 'lstm':
                decoder_output, (hidden, cell), attention_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_hidden = (hidden, cell)
            else:
                decoder_output, decoder_hidden, attention_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
            
            # Store prediction and attention weights
            outputs[:, t, :] = decoder_output
            attention_weights_history[:, t, :] = attention_weights
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = decoder_output.argmax(1)
            
            # Use teacher forcing or predicted token
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attention_weights_history
    
    def inference(self, source, source_length, sos_idx, eos_idx, max_length=50):
        """
        Perform inference (transliteration) on a source sequence.
        
        Args:
            source: Tensor of shape [batch_size, source_len]
            source_length: Length of each source sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            max_length: Maximum length of the target sequence
        
        Returns:
            predictions: List of predicted tokens
            attention_weights_history: Attention weights for visualization
        """
        batch_size = source.shape[0]
        max_src_len = source.shape[1]
        
        # Tensor to store attention weights
        attention_weights_history = torch.zeros(batch_size, max_length, max_src_len).to(self.device)
        
        # Encode the source sequence
        if self.encoder.cell_type == 'lstm':
            encoder_outputs, (hidden, cell) = self.encoder(source, source_length)
            decoder_hidden = (hidden, cell)
        else:
            encoder_outputs, hidden = self.encoder(source, source_length)
            decoder_hidden = hidden
        
        # First input to the decoder is the < SOS > token
        decoder_input = torch.tensor([[sos_idx]] * batch_size).to(self.device)
        
        # Lists to store predictions
        predictions = []
        
        # Decode one token at a time
        for t in range(max_length):
            # Pass through decoder with attention
            if self.decoder.cell_type == 'lstm':
                decoder_output, (hidden, cell), attention_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_hidden = (hidden, cell)
            else:
                decoder_output, decoder_hidden, attention_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
            
            # Store attention weights
            attention_weights_history[:, t, :] = attention_weights
            
            # Get the highest predicted token
            top1 = decoder_output.argmax(1)
            
            # Add prediction to list
            predictions.append(top1.item())
            
            # Stop if EOS token is predicted
            if top1.item() == eos_idx:
                break
            
            # Use predicted token as next input
            decoder_input = top1.unsqueeze(1)
        
        return predictions, attention_weights_history[:, :t+1, :]  # Return only valid time steps

# Helper function to visualize attention weights
def plot_attention(attention, source_chars, target_chars, title="Attention Weights"):
    """
    Visualize attention weights between source and target characters.
    
    Args:
        attention: Attention weights of shape [target_len, source_len]
        source_chars: List of source characters
        target_chars: List of target characters
        title: Title of the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=source_chars, yticklabels=target_chars, cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Source Characters")
    plt.ylabel("Target Characters")
    plt.tight_layout()
    
    return plt