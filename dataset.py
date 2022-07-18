from pathlib import Path
from PIL.Image import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd


class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, train_size):
        super(OCRDataset, self).__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.train_size = train_size
        transform_list = [transforms.Grayscale(1),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)

    @classmethod
    def from_csv(cls, filename, image_paths_column=None, labels_column=None, train_size=0.8):
        dataframe = pd.read_csv(filename)
        if image_paths_column is not None and labels_column is not None:
            image_paths = dataframe[image_paths_column]
            labels = dataframe[labels_column]
        else:
            image_paths = dataframe.iloc[:, 0]
            labels = dataframe.iloc[:, 1]
        return cls(image_paths, labels, train_size)

    @classmethod
    def from_matching_image_files(cls, directory, patterns, recursive, train_size=0.8):
        path = Path(directory)
        matching_image_paths = []
        prepend = "" if not recursive else "**/"
        for pattern in patterns:
            for x in path.glob(prepend + pattern):
                matching_image_paths.append(str(x))
        labels = []
        for file in matching_image_paths:
            with open(f'{file[:-4]}.txt') as reader:
                labels.append(reader.readlines()[0])
        return cls(matching_image_paths, labels, train_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        item = {"img": img, "idx": index, "label": label}
        return item
