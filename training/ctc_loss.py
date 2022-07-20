import torch
import math


class CustomCTCLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

    def forward(self, logits, labels, prediction_sizes, target_sizes):
        loss = self.ctc_loss(logits, labels, prediction_sizes, target_sizes)
        loss = self.sanitize(loss)
        return self.debug(loss, logits, labels, prediction_sizes, target_sizes)

    def sanitize(self, loss):
        eps = 1e-7
        if abs(loss.item() - float('inf')) < eps:
            return torch.zeros_like(loss)
        if math.isnan(loss.item()):
            return torch.zeros_like(loss)
        return loss

    def debug(self, loss, logits, labels,
              prediction_sizes, target_sizes):
        if math.isnan(loss.item()):
            print("loss:", loss)
            print("logits:", logits)
            print("labels:", labels)
            print("prediction_sizes:", prediction_sizes)
            print("target_sizes:", target_sizes)
            raise Exception("NaN loss obtained. But why?")
        return loss
