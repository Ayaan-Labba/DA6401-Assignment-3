import torch
import pandas as pd
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set seeds for reproducibility in Python, NumPy and PyTorch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class TransliterationDataset(Dataset):
    def __init__(self, data_path, max_len=50, lowercase=True):
        """
        Dataset for transliteration task
        
        Args:
            data_path: Path to the dataset
            max_len: Maximum length of sequences (including SOS and EOS tokens)
            lowercase: Whether to lowercase the Latin script text
        """

        # Read file
        self.data = pd.read_csv(data_path, sep='\t', header=None, dtype=str)
        
        # Clean the data
        self.data.dropna(inplace=True)
        self.data[0] = self.data[0].str.strip()  # Target (Malayalam)
        self.data[1] = self.data[1].str.strip()  # Source (Latin)
        
        # Apply lowercase
        if lowercase:
            self.data[1] = self.data[1].str.lower()
        
        self.max_len = max_len
        
        # Build vocabulary for source and target languages
        self.source_chars = set()
        self.target_chars = set()
        
        for _, row in self.data.iterrows():
            for char in row[1]:  # source (Latin)
                self.source_chars.add(char)
            for char in row[0]:  # target (Devanagari)
                self.target_chars.add(char)
        
        # Add special tokens
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        
        special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            self.source_chars.add(token)
            self.target_chars.add(token)
        
        # Create character to index mappings - ensure special tokens have fixed positions
        self.source_char_to_idx = {self.pad_token: 0, self.sos_token: 1, self.eos_token: 2, self.unk_token: 3}
        self.target_char_to_idx = {self.pad_token: 0, self.sos_token: 1, self.eos_token: 2, self.unk_token: 3}
        
        # Add remaining characters
        idx = 4
        for char in sorted(self.source_chars):
            if char not in self.source_char_to_idx:
                self.source_char_to_idx[char] = idx
                idx += 1
                
        idx = 4
        for char in sorted(self.target_chars):
            if char not in self.target_char_to_idx:
                self.target_char_to_idx[char] = idx
                idx += 1
        
        # Create index to character mappings
        self.source_idx_to_char = {idx: char for char, idx in self.source_char_to_idx.items()}
        self.target_idx_to_char = {idx: char for char, idx in self.target_char_to_idx.items()}
        
        # Save vocabulary sizes
        self.source_vocab_size = len(self.source_char_to_idx)
        self.target_vocab_size = len(self.target_char_to_idx)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source_text = self.data.iloc[idx, 1]  # Latin
        target_text = self.data.iloc[idx, 0]  # Malayalam
        
        # Convert to indices and add SOS/EOS tokens
        source_indices = [self.source_char_to_idx[self.sos_token]] + \
                        [self.source_char_to_idx.get(char, self.source_char_to_idx[self.unk_token]) for char in source_text] + \
                        [self.source_char_to_idx[self.eos_token]]
        
        target_indices = [self.target_char_to_idx[self.sos_token]] + \
                        [self.target_char_to_idx.get(char, self.target_char_to_idx[self.unk_token]) for char in target_text] + \
                        [self.target_char_to_idx[self.eos_token]]
        
        # Truncate if too long (keep SOS and add EOS)
        if len(source_indices) > self.max_len:
            source_indices = source_indices[:self.max_len-1] + [self.source_char_to_idx[self.eos_token]]
            
        if len(target_indices) > self.max_len:
            target_indices = target_indices[:self.max_len-1] + [self.target_char_to_idx[self.eos_token]]
        
        # Create tensors
        source_tensor = torch.LongTensor(source_indices)
        target_tensor = torch.LongTensor(target_indices)
        
        return {
            'source': source_tensor,
            'target': target_tensor,
            'source_text': source_text,
            'target_text': target_text,
            'source_length': len(source_indices),
            'target_length': len(target_indices)
        }
    
    def get_vocab_size(self, which='source'):
        """Get vocabulary size for source or target"""
        if which == 'source':
            return self.source_vocab_size
        else:
            return self.target_vocab_size
    
    def decode_indices(self, indices, which='target'):
        """
        Decode indices back to text
        
        Args:
            indices: List or tensor of indices
            which: 'source' or 'target'
        
        Returns:
            Decoded text
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
            
        idx_to_char = self.source_idx_to_char if which == 'source' else self.target_idx_to_char
        
        # Remove special tokens and decode
        text = []
        for idx in indices:
            char = idx_to_char[idx]
            if char in [self.pad_token, self.sos_token, self.eos_token]:
                if char == self.eos_token:
                    break  # Stop at EOS
                continue  # Skip other special tokens
            text.append(char)
            
        return ''.join(text)

def collate_fn(batch):
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary with batched tensors
    """
    # Sort batch by source sequence length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['source_length'], reverse=True)
    
    # Extract items
    source = [item['source'] for item in batch]
    target = [item['target'] for item in batch]
    source_length = [item['source_length'] for item in batch]
    target_length = [item['target_length'] for item in batch]
    
    # Pad sequences
    source_padded = pad_sequence(source, batch_first=True, padding_value=0)  # Use PAD token index (0)
    target_padded = pad_sequence(target, batch_first=True, padding_value=0)  # Use PAD token index (0)
    
    # Store original texts for evaluation
    source_text = [item['source_text'] for item in batch]
    target_text = [item['target_text'] for item in batch]
    
    return {
        'source': source_padded,
        'target': target_padded,
        'source_length': torch.LongTensor(source_length),
        'target_length': torch.LongTensor(target_length),
        'source_text': source_text,
        'target_text': target_text
    }