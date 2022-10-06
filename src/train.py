import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import *
from tqdm.auto import tqdm

from dataset import load_dataset
from model import VGG16
from config import Variables

class ImageClassifier:
    def __init__(self, model, dataloaders, validate=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(model.parameters(), lr=Variables.LR)
        self.loaders = dataloaders
        self.validate = validate
        self.step = 0
        self.log = {

        }

    def validation(self):
        self.model.eval()
        ground_truth = []
        predicate = []
        with torch.no_grad():
            for images, labels in tqdm(self.loaders['val'], desc="validating ..."):
                outputs = self.model(images.to(self.device))
                _, predicted = torch.max(outputs, 1)

                loss = self.criterion(outputs, labels)
                acc = accuracy_score(ground_truth, predicate)
                ground_truth.extend(labels.cpu())
                predicate.extend(predicted.cpu())

        print("accuracy score :", acc, "loss :", loss)
        return (ground_truth, predicate)

    def train(self):
        print("-- Start Training --")
        for epoch in range(10):
            self.model.train()
            running_loss = 0.0

            # with torch.set_grad_enabled(True):
            for batch, data in enumerate(tqdm(self.loaders['train'])):
                inputs, labels = [d.to(self.device) for d in data]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(outputs.shape)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                self.step += 1
            print(f'[{epoch + 1}, {batch + 1}] loss: {running_loss / (batch+1):.3f}')
            running_loss = 0.0

            if self.validate:
                self.validation()  

        print('-- Finished Training --')


if __name__ == "__main__":
    model = VGG16()
    datasets = {
        split: load_dataset(split) for split in Variables.SPLIT
    }
    dataloaders = {
        split: DataLoader(datasets[split], batch_size=Variables.BATCH_SIZE, shuffle=False)
        for split in datasets
    }

    clf = ImageClassifier(model, dataloaders)
    clf.train()