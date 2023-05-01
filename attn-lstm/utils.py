import torch

##########################################################FUNCTIONS FOR TRAIN.PY
def adjust_learning_rate(optimizer, shrink_factor):
    '''
    Shrinks the learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk
    :param shrink_factor: factor in interval (0,1) to multiply learning rate with
    '''

    print('\nDECAYING learning rate.')
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*shrink_factor
    print('The new learning rate is %f\n' %(optimizer.param_groups[0]['lr'],))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.resnet()

    def resnet(self):
        self.val = 0
        self.avg = 0
        self.sum_ = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum_ += val*n
        self.count +=  n
        self.avg = self.sum_ / self.count
        

def clip_gradient(optimizer, grad_clip):
    '''
    clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    '''

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy(scores, targets, k):
    '''
    computes the top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    '''

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

    
def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, save_dir):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, str(save_dir / filename))
    
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, str(save_dir / f"BEST_{filename}"))

def convert_tokens_to_words(tokens, inverse_wordmap):
    return [inverse_wordmap[token] for token in tokens]


