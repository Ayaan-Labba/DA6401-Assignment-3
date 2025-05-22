import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
import os

# Path to a font that supports Malayalam
MALAYALAM_FONT_PATH = "D:/Ayaan/IITM/Courses/Sem 8/DA6401/Noto_Sans_Malayalam/Noto_Sans_Malayalam/NotoSansMalayalam-VariableFont_wdth,wght.ttf"

def generate_attention_heatmaps(model, device, test_dataset, test_dataloader, num_samples=10):
    """
    Generate attention heatmaps for test samples and display them in a 3x3 grid format.
    
    Args:
        model: Trained attention-based Seq2Seq model
        device: Device to run the model on
        test_dataset: Test dataset
        test_dataloader: Test dataloader
        num_samples: Number of samples to visualize (default: 10)
    """
    model.eval()
    
    # Collect sample data
    samples = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            if sample_count >= num_samples:
                break
                
            source_texts = batch['source_text']
            target_texts = batch['target_text']
            
            for i, source_text in enumerate(source_texts):
                if sample_count >= num_samples:
                    break
                    
                # Get prediction and attention weights
                pred_text, attention_weights = transliterate_with_attention(
                    model, device, test_dataset, source_text
                )
                
                samples.append({
                    'source': source_text,
                    'target': target_texts[i],
                    'prediction': pred_text,
                    'attention': attention_weights
                })
                
                sample_count += 1
    
    # Create visualization with subplots arranged in a way that fits 10 samples
    # We'll use a 4x3 grid (12 subplots) but only use 10
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    fig.suptitle('Attention Heatmaps for Test Samples', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Load and set Malayalam font if available
    if os.path.exists(MALAYALAM_FONT_PATH):
        font_prop = fm.FontProperties(fname=MALAYALAM_FONT_PATH)
    else:
        font_prop = None  

    for idx, sample in enumerate(samples):
        if idx >= 12:  # Safety check
            break
            
        ax = axes[idx]
        
        source_chars = list(sample['source'])
        target_chars = list(sample['prediction']) if sample['prediction'] else ['']
        attention = sample['attention']
        
        # Handle empty predictions
        if len(target_chars) == 0 or attention is None:
            ax.text(0.5, 0.5, 'No prediction', ha='center', va='center', transform=ax.transAxes, fontproperties=font_prop)
            ax.set_title(f'Sample {idx+1}: {sample["source"]} → (empty)', fontsize=10, fontproperties=font_prop)
            ax.axis('off')
            continue
        
        # Trim attention matrix to match actual lengths
        attention_trimmed = attention[:len(target_chars), :len(source_chars)]
        
        # Create heatmap
        im = ax.imshow(attention_trimmed, cmap='Blues', aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(source_chars)))
        ax.set_yticks(range(len(target_chars)))
        ax.set_xticklabels(source_chars, fontsize=8, fontproperties=font_prop)
        ax.set_yticklabels(target_chars, fontsize=8, fontproperties=font_prop)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.ax.tick_params(labelsize=6)
        
        # Set title with source and prediction
        title = f'Sample {idx+1}'
        if len(sample['source']) > 15:
            title += f'\n{sample["source"][:15]}...'
        else:
            title += f'\n{sample["source"]}'
            
        if len(sample['prediction']) > 15:
            title += f' → {sample["prediction"][:15]}...'
        else:
            title += f' → {sample["prediction"]}'
            
        ax.set_title(title, fontsize=9, pad=10, fontproperties=font_prop)
        ax.set_xlabel('Source Characters', fontsize=8, fontproperties=font_prop)
        ax.set_ylabel('Target Characters', fontsize=8, fontproperties=font_prop)
        
        # Add grid for better readability
        ax.set_xticks(np.arange(-0.5, len(source_chars), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(target_chars), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Hide unused subplots
    for idx in range(num_samples, 12):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_attention/attention_heatmaps_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

def transliterate_with_attention(model, device, dataset, source_text, max_length=50):
    """
    Transliterate a source text and return both the prediction and attention weights.
    
    Args:
        model: Trained attention-based Seq2Seq model
        device: Device to run the model on  
        dataset: TransliterationDataset
        source_text: Source text to transliterate
        max_length: Maximum length of the target sequence
    
    Returns:
        target_text: Transliterated text
        attention_weights: Attention weights matrix [target_len, source_len]
    """
    model.eval()
    
    # Convert source text to indices
    source_indices = [dataset.source_char_to_idx['<SOS>']] + \
                     [dataset.source_char_to_idx.get(char, dataset.source_char_to_idx['<UNK>']) for char in source_text] + \
                     [dataset.source_char_to_idx['<EOS>']]
    
    # Create tensor
    source_tensor = torch.LongTensor(source_indices).unsqueeze(0).to(device)
    source_length = torch.LongTensor([len(source_indices)])
    
    # Get SOS and EOS tokens
    sos_idx = dataset.target_char_to_idx['<SOS>']
    eos_idx = dataset.target_char_to_idx['<EOS>']
    
    with torch.no_grad():
        # Use the model's inference method to get predictions and attention
        predicted_indices, attention_weights = model.inference(
            source_tensor, source_length, sos_idx, eos_idx, max_length
        )
        
        # Convert indices to characters
        target_chars = []
        for idx in predicted_indices:
            if idx == eos_idx:
                break
            if idx != sos_idx:
                target_chars.append(dataset.target_idx_to_char[idx])
        
        # Join characters to form target text
        target_text = ''.join(target_chars)
        
        # Convert attention weights to numpy for visualization
        if attention_weights.dim() == 3:  # [batch_size, target_len, source_len]
            attention_np = attention_weights.squeeze(0).cpu().numpy()
        else:
            attention_np = attention_weights.cpu().numpy()
    
    return target_text, attention_np


def create_individual_attention_plots(model, device, test_dataset, test_dataloader, num_samples=5):
    """
    Create individual attention heatmap plots for better visibility.
    
    Args:
        model: Trained attention-based Seq2Seq model
        device: Device to run the model on
        test_dataset: Test dataset
        test_dataloader: Test dataloader
        num_samples: Number of individual plots to create
    """
    model.eval()
    
    # Collect sample data
    samples = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            if sample_count >= num_samples:
                break
                
            source_texts = batch['source_text']
            target_texts = batch['target_text']
            
            for i, source_text in enumerate(source_texts):
                if sample_count >= num_samples:
                    break
                    
                # Get prediction and attention weights
                pred_text, attention_weights = transliterate_with_attention(
                    model, device, test_dataset, source_text
                )
                
                samples.append({
                    'source': source_text,
                    'target': target_texts[i],
                    'prediction': pred_text,
                    'attention': attention_weights
                })
                
                sample_count += 1
    
    # Load and set Malayalam font if available
    if os.path.exists(MALAYALAM_FONT_PATH):
        font_prop = fm.FontProperties(fname=MALAYALAM_FONT_PATH)
    else:
        font_prop = None

    # Create individual plots
    for idx, sample in enumerate(samples):
        plt.figure(figsize=(12, 8))
        
        source_chars = list(sample['source'])
        target_chars = list(sample['prediction']) if sample['prediction'] else ['']
        attention = sample['attention']
        
        if len(target_chars) == 0 or attention is None:
            plt.text(0.5, 0.5, 'No prediction available', ha='center', va='center', fontsize=16, fontproperties=font_prop)
            plt.title(f'Sample {idx+1}: {sample["source"]} (No Prediction)', fontproperties=font_prop)
            plt.axis('off')
            plt.savefig(f'predictions_attention/attention_heatmap_sample_{idx+1}.png', dpi=300, bbox_inches='tight')
            continue
        
        # Trim attention matrix to match actual lengths
        attention_trimmed = attention[:len(target_chars), :len(source_chars)]
        
        # Create heatmap
        sns.heatmap(attention_trimmed, 
                   xticklabels=source_chars, 
                   yticklabels=target_chars,
                   cmap='Blues', 
                   annot=True, 
                   fmt='.2f',
                   square=True,
                   cbar_kws={'label': 'Attention Weight'},
                   annot_kws={'fontproperties': font_prop} if font_prop else None)
        
        plt.title(f'Attention Heatmap - Sample {idx+1}\n'
                 f'Source: {sample["source"]} → Prediction: {sample["prediction"]}\n'
                 f'Target: {sample["target"]} | Match: {"Correct" if sample["prediction"] == sample["target"] else "Incorrect"}',
                 fontsize=14, pad=20, fontproperties=font_prop)
        
        plt.xlabel('Source Characters', fontsize=12, fontproperties=font_prop)
        plt.ylabel('Target Characters', fontsize=12, fontproperties=font_prop)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'predictions_attention/attention_heatmap_sample_{idx+1}.png', dpi=300, bbox_inches='tight')
        plt.show()