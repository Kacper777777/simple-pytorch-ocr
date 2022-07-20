import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, inp):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(inp)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self._num_classes = num_classes

        cnn = nn.Sequential()

        cnn.add_module('conv0', nn.Conv2d(3, 32, 3, 1, 1))
        cnn.add_module('batchnorm0', nn.BatchNorm2d(32))
        cnn.add_module('relu0', nn.ReLU(True))

        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))

        cnn.add_module('conv1', nn.Conv2d(32, 64, 3, 1, 1))
        cnn.add_module('batchnorm1', nn.BatchNorm2d(64))
        cnn.add_module('relu1', nn.ReLU(True))

        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(128, 64, 64),
            BidirectionalLSTM(64, 64, num_classes))

    def forward(self, inp):
        # conv features
        conv = self.cnn(inp)
        b, c, h, w = conv.size()
        conv = conv.reshape((b, h*c, w))
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c*h]
        # rnn features
        output = self.rnn(conv)
        output = output.transpose(1, 0)  # Tbh to bth
        return output
