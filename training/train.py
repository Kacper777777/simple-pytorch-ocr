import argparse
import json
import os
import torch
from torch.utils.data import random_split
from model import CRNN
from training.ctc_loss import CustomCTCLoss
from trainer import OCRTrainer
from utils import load_data


def parse_args():
    argparser = argparse.ArgumentParser(description="Train OCR model.")
    argparser.add_argument("-c", "--conf", help="Path to the configuration file.")
    args = argparser.parse_args()
    return args


def main(args):
    # Read the configuration file
    config_path = args.conf
    config_path = "/config.json"
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    train_config = config["training"]

    # Create the dataset
    dataset = load_data(train_config)

    train_split = int(train_config["train_size"] * len(dataset))
    val_split = len(dataset) - train_split

    data_train, data_val = torch.utils.data.random_split(dataset, [train_split, val_split])

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=train_config["batch_size"],
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=train_config["batch_size"])

    # indices = list(range(len(dataset)))
    # train_indices = indices[:train_config["train_size"]]
    # val_indices = indices[train_config["train_size"]:]
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=train_config["batch_size"], sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(dataset, batch_size=train_config["batch_size"], sampler=val_sampler)

    print(f"""Training data size: {train_split}""")
    print(f"""Validation data size: {val_split}""")

    model = CRNN(num_classes=len(config["alphabet"]))
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    criterion = CustomCTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])

    savepath = os.path.join(train_config["model_save_dir"], train_config["model_name"])
    os.makedirs(train_config["model_save_dir"], exist_ok=True)
    os.makedirs(train_config["log_dir"], exist_ok=True)

    trainer = OCRTrainer(criterion=criterion, optimizer=optimizer, schedule=False,
                         epochs=train_config["epochs"], batch_size=train_config["batch_size"],
                         resume=train_config["resume"], savepath=savepath,
                         log_dir=train_config["log_dir"], log_filename=train_config["log_filename"],
                         alphabet=config["alphabet"])
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
