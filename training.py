
import torch
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Training function
def train(model, device, dataloader, optimizer, criterion, clip=1.0, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        # Move batch to device
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        source_length = batch['source_length']
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(source, source_length, target, teacher_forcing_ratio)
        
        # Reshape output and target for loss calculation
        # output: [batch_size, target_len, output_size]
        # target: [batch_size, target_len]
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Skip the first timestep (<SOS>)
        target = target[:, 1:].reshape(-1)  # Skip the first timestep (<SOS>)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backpropagation
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

# Evaluation function
def evaluate(model, device, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_length = batch['source_length']
            
            # Forward pass
            output = model(source, source_length, target, teacher_forcing_ratio=0)
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)  # Skip the first timestep (<SOS>)
            target = target[:, 1:].reshape(-1)  # Skip the first timestep (<SOS>)
            
            # Calculate loss
            loss = criterion(output, target)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def transliterate(model, device, dataset, source_text, max_length=50):
    """
    Transliterate a source text to the target script.
    
    Args:
        model: Trained Seq2Seq model
        device: Device to run the model on
        dataset: TransliterationDataset
        source_text: Source text to transliterate
        max_length: Maximum length of the target sequence
    
    Returns:
        target_text: Transliterated text
    """
    model.eval()
    
    # Convert source text to indices
    source_indices = [dataset.source_char_to_idx['<SOS>']] + \
                     [dataset.source_char_to_idx[char] for char in source_text] + \
                     [dataset.source_char_to_idx['<EOS>']]
    
    # Create tensor
    source_tensor = torch.LongTensor(source_indices).unsqueeze(0).to(device)
    source_length = torch.LongTensor([len(source_indices)])
    
    # Get SOS and EOS tokens
    sos_idx = dataset.target_char_to_idx['<SOS>']
    eos_idx = dataset.target_char_to_idx['<EOS>']
    
    with torch.no_grad():
        # Use the model's inference method
        predicted_indices = model.inference(source_tensor, source_length, sos_idx, eos_idx, max_length)
    
    # Convert indices to characters
    target_chars = [dataset.target_idx_to_char[idx] for idx in predicted_indices 
                   if idx != sos_idx and idx != eos_idx]
    
    # Join characters to form target text
    target_text = ''.join(target_chars)
    
    return target_text

# Calculate the number of correct predictions
def calculate_accuracy(model, device, test_dataloader, dataset):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            source_texts = batch['source_text']
            target_texts = batch['target_text']
            
            for i, source_text in enumerate(source_texts):
                pred_text = transliterate(model, device, dataset, source_text)
                predictions.append((source_text, pred_text, target_texts[i]))
                
                if pred_text == target_texts[i]:
                    correct += 1
                total += 1
    
    accuracy = correct / total
    return accuracy, predictions

# # Inference function
# def transliterate(model, device, dataset, source_text, max_length=50):
#     """
#     Transliterate a source text to the target script.
    
#     Args:
#         model: Trained Seq2Seq model
#         dataset: TransliterationDataset
#         source_text: Source text to transliterate
#         max_length: Maximum length of the target sequence
    
#     Returns:
#         target_text: Transliterated text
#     """
#     model.eval()
    
#     # Convert source text to indices
#     source_indices = [dataset.source_char_to_idx['<SOS>']] + \
#                      [dataset.source_char_to_idx[char] for char in source_text] + \
#                      [dataset.source_char_to_idx['<EOS>']]
    
#     # Create tensor
#     source_tensor = torch.LongTensor(source_indices).unsqueeze(0).to(device)
#     source_length = torch.LongTensor([len(source_indices)])
    
#     # Get SOS and EOS tokens
#     sos_idx = dataset.target_char_to_idx['<SOS>']
#     eos_idx = dataset.target_char_to_idx['<EOS>']
    
#     with torch.no_grad():
#         # Encode the source sequence
#         if model.encoder.cell_type == 'lstm':
#             encoder_outputs, (hidden, cell) = model.encoder(source_tensor, source_length)
#             decoder_hidden = (hidden, cell)
#         else:
#             encoder_outputs, hidden = model.encoder(source_tensor, source_length)
#             decoder_hidden = hidden
        
#         # First input to the decoder is the <SOS> token
#         decoder_input = torch.tensor([[sos_idx]]).to(device)
        
#         # List to store predictions
#         predictions = []
        
#         # Decode one token at a time
#         for _ in range(max_length):
#             # Pass through decoder
#             if model.decoder.cell_type == 'lstm':
#                 decoder_output, (hidden, cell) = model.decoder(decoder_input, decoder_hidden)
#                 decoder_hidden = (hidden, cell)
#             else:
#                 decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            
#             # Get the highest predicted token
#             top1 = decoder_output.argmax(1)
            
#             # Add prediction to list
#             predictions.append(top1.item())
            
#             # Stop if EOS token is predicted
#             if top1.item() == eos_idx:
#                 break
            
#             # Use predicted token as next input
#             decoder_input = top1.unsqueeze(1)
    
#     # Convert indices to characters
#     target_chars = [dataset.target_idx_to_char[idx] for idx in predictions 
#                    if idx != sos_idx and idx != eos_idx]
    
#     # Join characters to form target text
#     target_text = ''.join(target_chars)
    
#     return target_text