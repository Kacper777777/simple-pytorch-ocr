import torch
import numpy as np
from functools import wraps
from time import time as _timenow
from sys import stderr
from textdistance import levenshtein as lev
from training.dataset import OCRDataset


def time(f):
    @wraps(f)
    def _wrapped(*args, **kwargs):
        start = _timenow()
        result = f(*args, **kwargs)
        end = _timenow()
        print(f"[time] {f.__name__}: {end - start}", file=stderr)
        return result

    return _wrapped


def load_data(config):
    mode = config["data"]["mode"]
    if mode == "from_csv":
        return OCRDataset.from_csv(filename=config["data"]["csv_name"],
                                   train_size=config["train_size"])
    elif mode == "from_matching_image_files":
        print(f'xc {config["data"]["directory"]}')
        return OCRDataset.from_matching_image_files(directory=config["data"]["directory"],
                                                    patterns=config["data"]["patterns"],
                                                    recursive=config["data"]["recursive"],
                                                    train_size=config["train_size"])
    else:
        raise Exception("Incorrect mode.")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_file, patience=5, verbose=False, delta=0, best_score=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_file = save_file
        print(best_score)

    def __call__(self, val_loss, epoch, model, optimizer):
        score = -val_loss
        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "best": score
        }
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, state)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: ({self.best_score:.6f} {self.counter} out of {self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, state)
            self.counter = 0

    def save_checkpoint(self, val_loss, state):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(state, self.save_file)
        self.val_loss_min = val_loss


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0
        self.max = -1 * float("inf")
        self.min = float("inf")

    def add(self, element):
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)

    def compute(self):
        return float("inf") if self.count == 0 else self.total / self.count

    def __str__(self):
        return f"{self.name} (min, avg, max): ({self.min:.3lf}, {self.compute():.3lf}, {self.max:.3lf})"


class Eval:
    def char_accuracy(self, pair):
        words, truths = pair
        words, truths = ''.join(words), ''.join(truths)
        sum_edit_dists = lev.distance(words, truths)
        sum_gt_lengths = sum(map(len, truths))
        fraction = sum_edit_dists / sum_gt_lengths if sum_gt_lengths != 0 else 0
        percent = fraction * 100
        if 100.0 - percent < 0:
            return 0.0
        else:
            return 100.0 - percent

    def word_accuracy(self, pair):
        word, truth = pair
        if word == truth:
            return 1
        else:
            return 0


class OCRLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + "-"  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
        self.dict[''] = 0

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        '''
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
        '''
        length = []
        result = []
        for item in text:
            # item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                if char in self.dict:
                    index = self.dict[char]
                else:
                    index = 0
                result.append(index)

        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strings.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): Decoded texts.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, f"Text with length: {t.numel()} " \
                                        f"does not match declared length: {length}"
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), f"Texts with length: {t.numel()} " \
                                              f"does not match declared length: {length.sum()}"
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
