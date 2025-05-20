
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

# Basic Encoder class
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

# Basic Decoder class
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, n_layers=1, dropout=0, cell_type='lstm'):
        """
        Args:
            output_size: Size of the vocabulary
            embedding_size: Size of the embeddings
            hidden_size: Size of the hidden state
            n_layers: Number of layers in the RNN
            dropout: Dropout probability
            cell_type: Type of RNN cell ('rnn', 'lstm', 'gru')
        """
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        
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
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        """
        Args:
            x: Tensor of shape [batch_size, 1]
            hidden: Hidden state from the encoder
        
        Returns:
            output: Tensor of shape [batch_size, output_size]
            hidden: Updated hidden state
        """
        # Embedding
        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embedding_size]
        
        # Pass through RNN
        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            # Convert output to vocabulary distribution
            prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
            return prediction, (hidden, cell)
        else:
            output, hidden = self.rnn(embedded, hidden)
            # Convert output to vocabulary distribution
            prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
            return prediction, hidden

# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Ensure the encoder and decoder have the same hidden dimensions
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
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
        """
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Encode the source sequence
        if self.encoder.cell_type == 'lstm':
            encoder_outputs, (hidden, cell) = self.encoder(source, source_length)
        else:
            encoder_outputs, hidden = self.encoder(source, source_length)
        
        # First input to the decoder is the <SOS> token
        decoder_input = target[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Pass the encoder's hidden state to the decoder
        if self.decoder.cell_type == 'lstm':
            decoder_hidden = (hidden, cell)
        else:
            decoder_hidden = hidden
        
        # Teacher forcing: use the ground-truth target as the next input
        # with probability teacher_forcing_ratio
        for t in range(1, target_len):
            # Pass through decoder
            if self.decoder.cell_type == 'lstm':
                decoder_output, (hidden, cell) = self.decoder(decoder_input, decoder_hidden)
                decoder_hidden = (hidden, cell)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Store prediction
            outputs[:, t, :] = decoder_output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = decoder_output.argmax(1)
            
            # Use teacher forcing or predicted token
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
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
        """
        batch_size = source.shape[0]
        
        # Encode the source sequence
        if self.encoder.cell_type == 'lstm':
            encoder_outputs, (hidden, cell) = self.encoder(source, source_length)
            decoder_hidden = (hidden, cell)
        else:
            encoder_outputs, hidden = self.encoder(source, source_length)
            decoder_hidden = hidden
        
        # First input to the decoder is the <SOS> token
        decoder_input = torch.tensor([[sos_idx]] * batch_size).to(self.device)
        
        # Lists to store predictions
        predictions = []
        
        # Decode one token at a time
        for _ in range(max_length):
            # Pass through decoder
            if self.decoder.cell_type == 'lstm':
                decoder_output, (hidden, cell) = self.decoder(decoder_input, decoder_hidden)
                decoder_hidden = (hidden, cell)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Get the highest predicted token
            top1 = decoder_output.argmax(1)
            
            # Add prediction to list
            predictions.append(top1.item())
            
            # Stop if EOS token is predicted
            if top1.item() == eos_idx:
                break
            
            # Use predicted token as next input
            decoder_input = top1.unsqueeze(1)
        
        return predictions