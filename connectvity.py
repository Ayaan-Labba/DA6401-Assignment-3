import torch
import json
import numpy as np
from attention_model import AttentionSeq2Seq
from dataset import TransliterationDataset

def generate_connectivity_visualization(model, dataset, device, test_dataloader):
    """
    Generate HTML visualization for attention connectivity
    
    Args:
        model: Trained AttentionSeq2Seq model
        dataset: TransliterationDataset instance
        device: torch device
        test_samples: List of source texts to visualize (if None, will use examples)
    
    Returns:
        HTML string for the connectivity visualization
    """
    
    # Collect sample data
    samples = []
    sample_count = 0
    num_samples = 3

    for batch in test_dataloader:
        if sample_count >= num_samples:
            break     
        source_texts = batch['source_text']
        
        for i, source_text in enumerate(source_texts):
            if sample_count >= num_samples:
                break
            
            samples.append(source_text)
            
            sample_count += 1
    
    # Get predictions and attention weights for each sample
    visualization_data = []
    
    model.eval()
    with torch.no_grad():
        for source_text in samples:
            try:
                # Convert source text to tensor
                source_indices = [dataset.source_char_to_idx.get('<SOS>', dataset.source_char_to_idx.get('< SOS >', 0))] + \
                               [dataset.source_char_to_idx.get(char, dataset.source_char_to_idx.get('<UNK>', 1)) 
                                for char in source_text] + \
                               [dataset.source_char_to_idx.get('<EOS>', dataset.source_char_to_idx.get('<EOS>', 2))]
                
                source_tensor = torch.LongTensor(source_indices).unsqueeze(0).to(device)
                source_length = torch.LongTensor([len(source_indices)])
                
                # Get prediction and attention weights
                sos_idx = dataset.target_char_to_idx.get('<SOS>', dataset.target_char_to_idx.get('< SOS >', 0))
                eos_idx = dataset.target_char_to_idx.get('<EOS>', dataset.target_char_to_idx.get('<EOS>', 2))
                
                # Check if model has inference method, otherwise use basic prediction
                if hasattr(model, 'inference'):
                    predicted_indices, attention_weights = model.inference(
                        source_tensor, source_length, sos_idx, eos_idx, max_length=50
                    )
                else:
                    # Fallback for models without inference method
                    outputs, attention_weights = model(source_tensor, source_tensor, source_length, source_length)
                    predicted_indices = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()
                
                # Convert prediction to text
                if hasattr(dataset, 'decode_indices'):
                    target_text = dataset.decode_indices(predicted_indices, 'target')
                else:
                    # Fallback decoding
                    target_text = ''.join([dataset.target_idx_to_char.get(idx, '<UNK>') 
                                         for idx in predicted_indices if idx not in [sos_idx, eos_idx]])
                
                # Process attention weights
                if attention_weights is not None:
                    if isinstance(attention_weights, torch.Tensor):
                        attention_weights = attention_weights.squeeze(0).cpu().numpy()
                    
                    # Ensure we have proper dimensions [target_len, source_len]
                    if len(attention_weights.shape) == 3:
                        attention_weights = attention_weights[0]  # Take first batch
                    
                    # Only keep attention for actual source characters (exclude special tokens)
                    actual_source_len = len(source_text)
                    actual_target_len = len(target_text)
                    
                    # Trim attention weights to actual lengths
                    if attention_weights.shape[1] > actual_source_len + 2:  # +2 for SOS/EOS
                        attention_trimmed = attention_weights[:actual_target_len, 1:actual_source_len+1]
                    else:
                        attention_trimmed = attention_weights[:actual_target_len, :actual_source_len]
                    
                    # Normalize attention weights
                    attention_trimmed = attention_trimmed / (attention_trimmed.sum(axis=1, keepdims=True) + 1e-8)
                else:
                    # Create dummy attention weights if not available
                    attention_trimmed = np.ones((len(target_text), len(source_text))) / len(source_text)
                
                visualization_data.append({
                    'source_text': source_text,
                    'target_text': target_text,
                    'source_chars': list(source_text),
                    'target_chars': list(target_text),
                    'attention_weights': attention_trimmed.tolist()
                })
                
            except Exception as e:
                print(f"Error processing sample '{source_text}': {str(e)}")
                continue
    
    if not visualization_data:
        print("No samples could be processed successfully")
        return "<html><body><h1>Error: No valid samples for visualization</h1></body></html>"
    
    # Generate HTML
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attention Connectivity Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .title {{
            text-align: center;
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 40px;
        }}
        
        .visualization-row {{
            margin-bottom: 50px;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }}
        
        .visualization-row:hover {{
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .row-title {{
            font-size: 1.3em;
            font-weight: 600;
            color: #495057;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .sequence-container {{
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .sequence {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .sequence-label {{
            font-weight: 600;
            color: #6c757d;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .char-container {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        
        .char {{
            padding: 12px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 1.2em;
            border: 2px solid transparent;
            min-width: 20px;
            text-align: center;
        }}
        
        .input-char {{
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: white;
        }}
        
        .output-char {{
            background: linear-gradient(135deg, #fd79a8, #e84393);
            color: white;
        }}
        
        .char:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }}
        
        .highlighted {{
            border: 3px solid #00b894 !important;
            box-shadow: 0 0 15px rgba(0, 184, 148, 0.5) !important;
            transform: translateY(-3px) scale(1.05) !important;
        }}
        
        .attention-info {{
            text-align: center;
            color: #6c757d;
            font-style: italic;
            margin-top: 15px;
            padding: 10px;
            background: rgba(108, 117, 125, 0.1);
            border-radius: 8px;
        }}
        
        .arrow {{
            font-size: 2em;
            color: #6c757d;
            margin: 0 20px;
        }}
        
        .attention-heatmap {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            border: 1px solid #dee2e6;
        }}
        
        .heatmap-title {{
            font-weight: 600;
            color: #495057;
            margin-bottom: 10px;
            text-align: center;
        }}
        
        .heatmap-grid {{
            display: grid;
            gap: 2px;
            justify-content: center;
            max-width: 100%;
            overflow-x: auto;
        }}
        
        .heatmap-cell {{
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            border-radius: 3px;
            color: white;
            font-weight: 600;
        }}
        
        @media (max-width: 768px) {{
            .sequence-container {{
                flex-direction: column;
                gap: 20px;
            }}
            
            .arrow {{
                transform: rotate(90deg);
                margin: 10px 0;
            }}
            
            .char-container {{
                max-width: 300px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">Attention Connectivity Visualization</h1>
        <p class="subtitle">Hover over output characters to see which input characters the model focuses on</p>
        
        {sample_rows}
    </div>

    <script>
        const attentionWeights = {attention_weights_json};

        function highlightAttention(sampleId, outputId) {{
            clearHighlights(sampleId);
            
            if (!attentionWeights[sampleId] || !attentionWeights[sampleId][outputId]) {{
                return;
            }}
            
            const weights = attentionWeights[sampleId][outputId];
            const inputChars = document.querySelectorAll(`#input-${{sampleId}} .char`);
            const infoDiv = document.getElementById(`info-${{sampleId}}`);
            
            weights.forEach((weight, index) => {{
                if (index < inputChars.length && weight > 0.01) {{
                    const opacity = Math.max(0.2, Math.min(1, weight * 2));
                    const intensity = Math.min(1, weight * 3);
                    
                    if (weight > 0.1) {{
                        inputChars[index].classList.add('highlighted');
                        inputChars[index].style.boxShadow = `0 0 ${{10 + intensity * 15}}px rgba(0, 184, 148, ${{opacity}})`;
                        inputChars[index].style.borderColor = `rgba(0, 184, 148, ${{intensity}})`;
                    }}
                }}
            }});
            
            const maxWeight = Math.max(...weights);
            const maxIndex = weights.indexOf(maxWeight);
            const focusChar = inputChars[maxIndex]?.dataset.char || '';
            infoDiv.textContent = `Strongest attention: "${{focusChar}}" (${{(maxWeight * 100).toFixed(1)}}%)`;
        }}

        function clearHighlights(sampleId) {{
            const inputChars = document.querySelectorAll(`#input-${{sampleId}} .char`);
            const infoDiv = document.getElementById(`info-${{sampleId}}`);
            
            inputChars.forEach(char => {{
                char.classList.remove('highlighted');
                char.style.boxShadow = '';
                char.style.borderColor = 'transparent';
            }});
            
            infoDiv.textContent = 'Hover over an output character to see attention weights';
        }}

        // Initialize event listeners
        for (let sampleId = 0; sampleId < {num_samples}; sampleId++) {{
            const outputChars = document.querySelectorAll(`#output-${{sampleId}} .char`);
            
            outputChars.forEach((char, index) => {{
                char.addEventListener('mouseenter', () => {{
                    highlightAttention(sampleId, index);
                }});
                
                char.addEventListener('mouseleave', () => {{
                    setTimeout(() => clearHighlights(sampleId), 100);
                }});
            }});
        }}

        // Add hover effects for all characters
        document.querySelectorAll('.char').forEach(char => {{
            char.addEventListener('mouseenter', function() {{
                if (!this.classList.contains('highlighted')) {{
                    this.style.transform = 'translateY(-3px) scale(1.05)';
                }}
            }});
            
            char.addEventListener('mouseleave', function() {{
                if (!this.classList.contains('highlighted')) {{
                    this.style.transform = '';
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    # Generate sample rows HTML
    sample_rows = ""
    attention_weights_dict = {}
    
    for i, data in enumerate(visualization_data):
        source_chars_html = ""
        for char in data['source_chars']:
            source_chars_html += f'<div class="char input-char" data-char="{char}">{char}</div>'
        
        target_chars_html = ""
        for j, char in enumerate(data['target_chars']):
            target_chars_html += f'<div class="char output-char" data-output="{j}" data-char="{char}">{char}</div>'
        
        # Generate attention heatmap
        heatmap_html = generate_heatmap_html(data['attention_weights'], data['source_chars'], data['target_chars'])
        
        sample_rows += f"""
        <div class="visualization-row">
            <div class="row-title">Sample {i+1}: "{data['source_text']}" → "{data['target_text']}"</div>
            <div class="sequence-container">
                <div class="sequence">
                    <div class="sequence-label">Input (Latin)</div>
                    <div class="char-container" id="input-{i}">
                        {source_chars_html}
                    </div>
                </div>
                <div class="arrow">→</div>
                <div class="sequence">
                    <div class="sequence-label">Output (Devanagari)</div>
                    <div class="char-container" id="output-{i}">
                        {target_chars_html}
                    </div>
                </div>
            </div>
            <div class="attention-info" id="info-{i}">Hover over an output character to see attention weights</div>
            {heatmap_html}
        </div>
        """
        
        # Store attention weights for JavaScript
        attention_weights_dict[i] = {}
        for j, weights in enumerate(data['attention_weights']):
            if j < len(data['target_chars']):
                attention_weights_dict[i][j] = weights
    
    # Fill in the template
    html_content = html_template.format(
        sample_rows=sample_rows,
        attention_weights_json=json.dumps(attention_weights_dict),
        num_samples=len(visualization_data)
    )
    
    return html_content

def generate_heatmap_html(attention_weights, source_chars, target_chars):
    """Generate HTML for attention heatmap visualization"""
    if not attention_weights or not source_chars or not target_chars:
        return ""
    
    # Ensure we have proper dimensions
    if len(attention_weights) != len(target_chars):
        return ""
    
    heatmap_html = '<div class="attention-heatmap">'
    heatmap_html += '<div class="heatmap-title">Attention Heatmap</div>'
    heatmap_html += f'<div class="heatmap-grid" style="grid-template-columns: 50px repeat({len(source_chars)}, 1fr);">'
    
    # Header row
    heatmap_html += '<div class="heatmap-cell" style="background: #6c757d;"></div>'
    for src_char in source_chars:
        heatmap_html += f'<div class="heatmap-cell" style="background: #74b9ff;">{src_char}</div>'
    
    # Data rows
    for i, (tgt_char, weights) in enumerate(zip(target_chars, attention_weights)):
        heatmap_html += f'<div class="heatmap-cell" style="background: #fd79a8;">{tgt_char}</div>'
        for j, weight in enumerate(weights[:len(source_chars)]):
            intensity = int(255 * (1 - weight))
            color = f'rgb({intensity}, 255, {intensity})'
            heatmap_html += f'<div class="heatmap-cell" style="background: {color}; color: black;" title="Attention: {weight:.3f}">{weight:.2f}</div>'
    
    heatmap_html += '</div></div>'
    return heatmap_html

def save_connectivity_visualization(model, dataset, device, output_path="connectivity_visualization.html", test_samples=None):
    """
    Generate and save the connectivity visualization HTML file
    
    Args:
        model: Trained AttentionSeq2Seq model
        dataset: TransliterationDataset instance
        device: torch device
        output_path: Path to save the HTML file
        test_samples: List of source texts to visualize
    """
    try:
        html_content = generate_connectivity_visualization(model, dataset, device, test_samples)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Connectivity visualization saved to {output_path}")
        return html_content
        
    except Exception as e:
        print(f"Error generating connectivity visualization: {str(e)}")
        return None

def batch_connectivity_visualization(model, dataset, device, test_loader, num_samples=5, output_path="connectivity_visualization.html"):
    """
    Generate connectivity visualization from a test data loader
    
    Args:
        model: Trained AttentionSeq2Seq model
        dataset: TransliterationDataset instance
        device: torch device
        test_loader: DataLoader with test samples
        num_samples: Number of samples to visualize
        output_path: Path to save the HTML file
    """
    test_samples = []
    
    try:
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            
            # Extract source text from batch
            if hasattr(dataset, 'decode_indices'):
                source_indices = batch[0][0].cpu().numpy()  # First sample in batch
                source_text = dataset.decode_indices(source_indices, 'source')
                # Remove special tokens
                source_text = source_text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
                if source_text:
                    test_samples.append(source_text)
            
        return save_connectivity_visualization(model, dataset, device, output_path, test_samples)
        
    except Exception as e:
        print(f"Error in batch connectivity visualization: {str(e)}")
        return None