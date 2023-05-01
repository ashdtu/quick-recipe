import torch
import numpy as np


def greedy_decoding(model, img_features_batched, sos_id, eos_id, pad_id, idx2word, max_len, device):
    """Performs greedy decoding for the caption generation.
    At each iteration model predicts the next word in the caption given the previously
    generated words and image features. For the next word we always take the most probable one.
    Arguments:
        model (torch.nn.Module): Transformer Decoder model which generates prediction for the next word
        img_features_padded (torch.Tensor): Image features generated by CNN encoder
            Stacked along 0-th dimension for each image in the mini-batch
        sos_id (int): Id of <start> token in the vocabulary
        eos_id (int): Id of <end> token in the vocabulary
        pad_id (int): Id of <pad> token in the vocabulary
        idx2word (dict): Mapping from ordinal number of token (i.e. class number) to the string of word
        max_len (int): Maximum length of the caption
        device (torch.device): Device on which to port used tensors
    Returns:
        generated_captions (list of str): Captions generated for each image in the batch
    """
    batch_size = img_features_batched.size(0)

    # Define the initial state of decoder input
    x_words = torch.Tensor([sos_id] + [pad_id] * (max_len - 1)).to(device).long()
    x_words = x_words.repeat(batch_size, 1)
    padd_mask = torch.Tensor([True] * max_len).to(device).bool()
    padd_mask = padd_mask.repeat(batch_size, 1)

    # Is each image from the batch decoded
    is_decoded = [False] * batch_size
    generated_captions = []
    for _ in range(batch_size):
        generated_captions.append([])

    for i in range(max_len - 1):
        # Update the padding masks
        padd_mask[:, i] = False

        # Get the model prediction for the next word
        y_pred_prob = model(x_words, img_features_batched, padd_mask)
        # Extract the prediction from the specific (next word) position of the target sequence
        y_pred_prob = y_pred_prob[torch.arange(batch_size), [i] * batch_size].clone()
        # Extract the most probable word
        y_pred = y_pred_prob.argmax(-1)
        
        for batch_idx in range(batch_size):
            if is_decoded[batch_idx]:
                continue
            # Add the generated word to the caption
            # print(f"y_pred: {y_pred} \n batch_idx: {batch_idx}")
            generated_captions[batch_idx].append(idx2word[str(y_pred[batch_idx].item())])
            if y_pred[batch_idx] == eos_id:
                # Caption has been fully generated for this image
                is_decoded[batch_idx] = True
            
        if np.all(is_decoded):
            break

        if i < (max_len - 1):   # We haven't reached maximum number of decoding steps
            # Update the input tokens for the next iteration
            x_words[torch.arange(batch_size), [i+1] * batch_size] = y_pred.view(-1)

    # Complete the caption for images which haven't been fully decoded
    for batch_idx in range(batch_size):
        if not is_decoded[batch_idx]:
            generated_captions[batch_idx].append(idx2word[str(eos_id)])

    # Clean the EOS symbol
    for caption in generated_captions:
        caption.remove("<end>")

    return generated_captions
