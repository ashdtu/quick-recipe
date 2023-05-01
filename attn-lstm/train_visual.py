import time
import torch.backends.cudnn as cudnn
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
# from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
# from models import Encoder, DecoderWithAttention
from models import DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from pathlib import Path

data_folder = 'data_output'  # folder with data files saved by create_input_files.py
data_name = 'youcook_attnlstm_visual'   # base name shared by data files
PROJ_DIR = Path.cwd().parent
DATA_DIR = PROJ_DIR / 'data'
LOG_DIR = PROJ_DIR / 'logs'
LOG_SAVE_DIR = LOG_DIR / data_name
LOG_SAVE_DIR.mkdir(parents=True, exist_ok=True) 
MODEL_DIR = PROJ_DIR / 'models'
MODEL_SAVE_DIR = MODEL_DIR / data_name

# Model parameters
# emb_dim = 512  # dimension of word embeddings
# attention_dim = 512  # dimension of attention linear layers
# decoder_dim = 512  # dimension of decoder RNN
emb_dim = 512  # dimension of word embeddings
attention_dim = 256  # dimension of attention linear layers
decoder_dim = 256  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder

# BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar
checkpoint = None  # path to checkpoint, None if none
top_k_acc = 3


def main():
    '''Training and Validation'''

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = f"{data_folder}/wordmap.json"
    print(word_map_file)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)


    # Initialize or load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       encoder_dim=32, # 512 -> 4,4,32
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        
        decoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, decoder.parameters()),lr=decoder_lr)

        # encoder = Encoder()
        encoder = None
        # encoder.fine_tune(fine_tune_encoder)
        # encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
        #                                      lr=encoder_lr) if fine_tune_encoder else None
        encoder_optimizer = None

    else:
        # Loading the checkpoint
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['blue-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        # if fine_tune_encoder is True and encoder_optimizer is None:
            # encoder.fine_tune(fine_tune_encoder)
            # encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
            #                                      lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    # encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom DataLoaders
    # normalizing with args most suited for Resnet-101, check torchvision docs for more info.
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std =[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        # Custom dataset
        CaptionDataset(str(DATA_DIR), data_name, 'TRAIN', transform=None),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    ) #If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, you can speed up the host to device transfer by enabling pin_memory.

    val_loader = DataLoader(
        CaptionDataset(str(DATA_DIR), data_name, 'VAL', transform=None),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # test_loader = DataLoader(
    #     CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
    #     batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    print(f'Training on {device}')

    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            # if encoder is being trained too then change its lr
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(
            train_loader=train_loader,
            encoder = encoder,
            decoder = decoder,
            criterion = criterion,
            encoder_optimizer = encoder_optimizer,
            decoder_optimizer = decoder_optimizer,
            epoch = epoch
        )

        # One epoch's validation
        recent_bleu4 = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion
        )

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # save checkpoint
        save_checkpoint(data_name,epoch,epochs_since_improvement,encoder,decoder,encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, str(MODEL_SAVE_DIR))


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    '''
    Performs one epoch of training

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param  encoder_optimizer: optimizer to update encoder's weights ( if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number 
    '''

    # Set both the models to train mode: This will enable all dropouts and batch_norm layers
    decoder.train()
    if encoder is not None:
        encoder.train()

    batch_time = AverageMeter() #forward prop + back prop time
    data_time  = AverageMeter() # data loading time
    losses = AverageMeter() # loss (per word decoded)
    top5accs = AverageMeter() # top 5 accuracy

    start = time.time()

    # Batches 
    for i, (imgs, caps, caplens) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        # imgs.size() = [batch_size, C, H, W] == [32,3,256,256]
        imgs = imgs.to(device)
        # print(f"imgs size: {imgs.size()}")
        # caps.size() = [batch_size, max_caption_len] == [32,100] 
        caps = caps.to(device)
        # print(f"caps size: {caps.size()}")
        # caplens.size() = [batch_size, 1] == [32,1] 
        caplens = caplens.to(device)
        # print(f"caplens size: {caplens.size()}")
        # Forward prop
        # imgs = encoder(imgs)
        # imgs.size() = [batch_size, 14,14,2048]

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        # print(f"scores size: {scores.size()}")
        # since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first = True)
        scores = scores.data
        targets= pack_padded_sequence(targets, decode_lengths, batch_first=True)
        targets = targets.data

        # Calculate the loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c*((1.-alphas.sum(dim=1)**2)).mean()

        # Back prop
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, top_k_acc)
        losses.update(loss.item(), sum(decode_lengths))
        

        # Keep track of metrics
        top5 = accuracy(scores, targets, top_k_acc) 
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-{k} Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                          top5=top5accs, k=top_k_acc))

def validate(val_loader, encoder, decoder, criterion):
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

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

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

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, top_k_acc)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-{k} Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs, k=top_k_acc))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder 
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypothesis
            _, preds = torch.max(scores_copy, dim = 2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-{k} ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4,
                k=top_k_acc,))

    return bleu4


if __name__ == '__main__':
    main()

            