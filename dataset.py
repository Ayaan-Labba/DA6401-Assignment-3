import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# Data preparation
class TransliterationDataset(Dataset):
    def __init__(self, data_path, max_len=50):
        """
        Dataset for transliteration task
        
        Args:
            data_path: Path to the dataset
            max_len: Maximum length of sequences
        """
        self.data = pd.read_csv(data_path, sep='\t', header=None, dtype=str)
        self.data.dropna(inplace=True)
        self.data[0] = self.data[0].str.strip()
        self.data[1] = self.data[1].str.strip()

        self.max_len = max_len
        
        # Build vocabulary for source and target languages
        self.source_chars = set()
        self.target_chars = set()
        
        for _, row in self.data.iterrows():
            for char in row[1]:  # source (Latin)
                self.source_chars.add(char)
            for char in row[0]:  # target (Malayalam)
                self.target_chars.add(char)
        
        # Add special tokens
        self.source_chars.add('<PAD>')
        self.source_chars.add('<SOS>')
        self.source_chars.add('<EOS>')
        self.target_chars.add('<PAD>')
        self.target_chars.add('<SOS>')
        self.target_chars.add('<EOS>')
        
        # Create character to index mappings
        self.source_char_to_idx = {char: idx for idx, char in enumerate(sorted(self.source_chars))}
        self.source_idx_to_char = {idx: char for char, idx in self.source_char_to_idx.items()}
        
        self.target_char_to_idx = {char: idx for idx, char in enumerate(sorted(self.target_chars))}
        self.target_idx_to_char = {idx: char for char, idx in self.target_char_to_idx.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source_text = self.data.iloc[idx, 1]
        target_text = self.data.iloc[idx, 0]
        
        # Convert to indices and add SOS/EOS tokens
        source_indices = [self.source_char_to_idx['<SOS>']] + \
                         [self.source_char_to_idx[char] for char in source_text] + \
                         [self.source_char_to_idx['<EOS>']]
        
        target_indices = [self.target_char_to_idx['<SOS>']] + \
                         [self.target_char_to_idx[char] for char in target_text] + \
                         [self.target_char_to_idx['<EOS>']]
        
        # Truncate if too long
        source_indices = source_indices[:self.max_len]
        target_indices = target_indices[:self.max_len]
        
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
        if which == 'source':
            return len(self.source_chars)
        else:
            return len(self.target_chars)

# Custom collate function for batching
def collate_fn(batch):
    # Sort batch by source sequence length (descending)
    batch = sorted(batch, key=lambda x: x['source_length'], reverse=True)
    
    source = [item['source'] for item in batch]
    target = [item['target'] for item in batch]
    source_length = [item['source_length'] for item in batch]
    target_length = [item['target_length'] for item in batch]
    
    # Pad sequences
    source_padded = pad_sequence(source, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target, batch_first=True, padding_value=0)
    
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