from collections import OrderedDict
import os
import logging
from tqdm import *
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import AverageMeter, OCRLabelConverter, Eval, EarlyStopping


class OCRTrainer(object):
    def __init__(self, criterion, optimizer,
                 schedule, epochs, batch_size, resume, savepath,
                 log_dir, log_filename, alphabet):
        super(OCRTrainer, self).__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedule = schedule
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs
        self.batch_size = batch_size

        self.savepath = savepath #os.path.join(savepath, "best.ckpt")
        self.model = None
        self.epoch = 0
        self.count = 1
        self.best_score = None
        if resume and os.path.exists(self.savepath):
            self.checkpoint = torch.load(self.savepath)
            self.epoch = self.checkpoint["epoch"]
            self.count = self.epoch
            self.best_score = self.checkpoint["best"]
            self.load()
        else:
            print("Checkpoint does not exist.")

        self.log_dir = log_dir
        self.log_filename = log_filename
        self.saver = EarlyStopping(self.savepath, patience=15, verbose=True, best_score=self.best_score)
        self.converter = OCRLabelConverter(alphabet)
        self.evaluator = Eval()
        self.init_meters()

    def fit(self, model, train_data_loader, val_data_loader):
        logging.basicConfig(filename=f"""{self.log_dir}/{self.log_filename}.csv""", level=logging.INFO)

        self.model = model

        for epoch in range(self.epoch, self.epochs):
            train_result = self.run_epoch(train_data_loader)
            val_result = self.run_epoch(val_data_loader, validation=True)
            self.count = epoch
            info = f"{epoch}, " \
                   f"{train_result['train_loss']:.6f}, {val_result['val_loss']:.6f}, " \
                   f"{train_result['train_ca']:.6f}, {val_result['val_ca']:.6f}, " \
                   f"{train_result['train_wa']:.6f}, {val_result['val_wa']:.6f}"
            logging.info(info)
            self.val_loss = val_result['val_loss']
            print(self.val_loss)
            if self.savepath:
                self.save(epoch)
            if self.saver.early_stop:
                print("Early stopping.")
                break

    def load(self):
        print(f"""Loading checkpoint at {self.savepath} trained for {self.checkpoint["epoch"]} epochs.""")
        self.model.load_state_dict(self.checkpoint["state_dict"])
        if "opt_state_dict" in self.checkpoint.keys():
            print("Loading optimizer.")
            self.optimizer.load_state_dict(self.checkpoint["opt_state_dict"])

    def save(self, epoch):
        self.saver(self.val_loss, epoch, self.model, self.optimizer)

    def init_meters(self):
        self.avg_train_loss = AverageMeter("Train loss")
        self.avg_train_char_accuracy = AverageMeter("Train character accuracy")
        self.avg_train_word_accuracy = AverageMeter("Train word accuracy")
        self.avg_val_loss = AverageMeter("Validation loss")
        self.avg_val_char_accuracy = AverageMeter("Validation character accuracy")
        self.avg_val_word_accuracy = AverageMeter("Validation word accuracy")

    def forward(self, x):
        logits = self.model(x)
        return logits.transpose(1, 0)

    def loss_fn(self, logits, targets, pred_sizes, target_sizes):
        loss = self.criterion(logits, targets, pred_sizes, target_sizes)
        return loss

    def step(self):
        clip_grad_norm_(self.model.parameters(), 0.05)
        self.optimizer.step()

    def schedule_lr(self):
        if self.schedule:
            self.scheduler.step()

    def _run_batch(self, batch, report_accuracy=False):
        input_, targets = batch["img"], batch["label"]
        targets, lengths = self.converter.encode(targets)
        logits = self.forward(input_)
        logits = logits.contiguous().cpu()
        logits = torch.nn.functional.log_softmax(logits, 2)
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        targets = targets.view(-1).contiguous()
        loss = self.loss_fn(logits, targets, pred_sizes, lengths)
        if report_accuracy:
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, pred_sizes.data, raw=False)
            ca = np.mean((list(map(self.evaluator.char_accuracy, list(zip(sim_preds, batch['label']))))))
            wa = np.mean((list(map(self.evaluator.word_accuracy, list(zip(sim_preds, batch['label']))))))
            return loss, ca, wa
        else:
            return loss

    def run_epoch(self, data_loader, validation=False):
        if not validation:
            pbar = tqdm(data_loader, desc=f"Epoch: {self.count}/{self.epochs} Training", leave=True)
            self.model.train()
        else:
            pbar = tqdm(data_loader, desc="Validating", leave=True)
            self.model.eval()

        outputs = []
        for batch_nb, batch in enumerate(pbar):
            if not validation:
                output = self.training_step(batch)
            else:
                output = self.validation_step(batch)
            pbar.set_postfix(output)
            outputs.append(output)

        self.schedule_lr()

        if not validation:
            result = self.train_end(outputs)
        else:
            result = self.validation_end(outputs)

        return result

    def training_step(self, batch):
        loss, ca, wa = self._run_batch(batch, report_accuracy=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        output = OrderedDict({
            "loss": abs(loss.item()),
            "train_ca": ca.item(),
            "train_wa": wa.item()
        })
        return output

    def validation_step(self, batch):
        loss, ca, wa = self._run_batch(batch, report_accuracy=True)
        output = OrderedDict({
            "val_loss": abs(loss.item()),
            "val_ca": ca.item(),
            "val_wa": wa.item()
        })
        return output

    def train_end(self, outputs):
        for output in outputs:
            self.avg_train_loss.add(output["loss"])
            self.avg_train_char_accuracy.add(output["train_ca"])
            self.avg_train_word_accuracy.add(output["train_wa"])

        train_loss_mean = abs(self.avg_train_loss.compute())
        train_ca_mean = self.avg_train_char_accuracy.compute()
        train_wa_mean = self.avg_train_word_accuracy.compute()

        result = {"train_loss": train_loss_mean, "train_ca": train_ca_mean,
                  "train_wa": train_wa_mean}
        return result

    def validation_end(self, outputs):
        for output in outputs:
            self.avg_val_loss.add(output["val_loss"])
            self.avg_val_char_accuracy.add(output["val_ca"])
            self.avg_val_word_accuracy.add(output["val_wa"])

        val_loss_mean = abs(self.avg_val_loss.compute())
        val_ca_mean = self.avg_val_char_accuracy.compute()
        val_wa_mean = self.avg_val_word_accuracy.compute()

        result = {"val_loss": val_loss_mean, "val_ca": val_ca_mean,
                  "val_wa": val_wa_mean}
        return result
