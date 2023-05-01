import json
import time

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from pathlib import Path

# Parameters
# data_folder = 'data_output'  # folder with data files saved by create_input_files.py
# data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
data_name = 'youcook_attnlstm_multimodal'
PROJ_DIR = Path.cwd().parent
DATA_DIR = PROJ_DIR / 'data'
LOG_DIR = PROJ_DIR / 'logs'
LOG_SAVE_DIR = LOG_DIR / data_name
LOG_SAVE_DIR.mkdir(parents=True, exist_ok=True) 
MODEL_DIR = PROJ_DIR / 'models'
checkpoint = str(MODEL_DIR / 'BEST_checkpoint_quick-recipe-multimodal.pth.tar')  # model checkpoint
word_map_file = 'data_output/wordmap.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead



# Normalization transform
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])


def validate(val_loader, encoder, decoder):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    total_true_pos = 0
    total_num_predicted = 0
    total_num_gold = 0
    summary_list = []
    predictions_list = []
    true_positive_list = []
    num_predicted_list = []
    num_gold_list = []


    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in tqdm(enumerate(val_loader)):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            scores = scores.data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            targets = targets.data

            # # Calculate loss
            # loss = criterion(scores, targets)

            # # Add doubly stochastic attention regularization
            # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            # losses.update(loss.item(), sum(decode_lengths))
            # top5 = accuracy(scores, targets, top_k_acc)
            # top5accs.update(top5, sum(decode_lengths))
            # batch_time.update(time.time() - start)

            start = time.time()

            # if i % print_freq == 0:
            #     print('Validation: [{0}/{1}]\t'
            #           'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Top-{k} Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
            #                                                                     loss=losses, top5=top5accs, k=top_k_acc))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder 
            summary_words = None
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                summary_words = [convert_tokens_to_words(cap, rev_word_map) for cap in img_captions][0]
                # print("Img captions", summary_words)
                
                references.append(img_captions)
            
            

            # Hypothesis
            _, preds = torch.max(scores_copy, dim = 2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            predicted_words = [convert_tokens_to_words(preds[0], rev_word_map) for pred in preds][0]
            # print("Preds", predicted_words)
            # print(f"Preds: {convert_tokens_to_words(preds[0], )}")
            hypotheses.extend(preds)

            true_pos = len(set(predicted_words) & set(summary_words))
            num_predicted = len(set(predicted_words))
            num_gold = len(set(summary_words))
            total_true_pos += true_pos
            total_num_predicted += num_predicted
            total_num_gold += num_gold
            summary_list.append(' '.join(summary_words))
            predictions_list.append(' '.join(predicted_words))
            true_positive_list.append(true_pos)
            num_predicted_list.append(num_predicted)
            num_gold_list.append(num_gold)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        results_df = pd.DataFrame({
                'summary': summary_list,
                'prediction': predictions_list,
                'true_positives': true_positive_list,
                'num_predicted': num_predicted_list,
                'num_gold': num_gold_list
                })
        results_df.to_csv(f"{str(LOG_SAVE_DIR)}/val_performance.csv", index=False)
        precision = round(total_true_pos / total_num_predicted, 2)
        recall = round(total_true_pos / total_num_gold, 2)
        f1 = round(2*(precision*recall) / (precision+recall), 2)

        print(f"Total true positives (predicted words overlap with gold words): {total_true_pos}")
        print(f"Total predicted words: {total_num_predicted}")
        print(f"Total gold words: {total_num_gold}")
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
        # print(
        #     '\n * LOSS - {loss.avg:.3f}, TOP-{k} ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
        #         loss=losses,
        #         top5=top5accs,
        #         bleu=bleu4,
        #         k=top_k_acc,))

    return bleu4


if __name__ == '__main__':
    # beam_size = int(input('Enter Beam Size:'))
    # beam_size = 3    
    # print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
    # Load model
    checkpoint = torch.load(checkpoint)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    if encoder is not None:
        encoder = encoder.to(device)
        encoder.eval()

    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)
    batch_size = 1
    workers = 1

    val_loader = DataLoader(
        CaptionDatasetMultimodal(str(DATA_DIR), data_name, 'TEST', transform=None),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    validate(val_loader, encoder, decoder)