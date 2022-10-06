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
        self.log_step = 1
        self.step = 0
        self.log = {
            "train_acc": [],
            "train_loss": [],
            "train_step": [],
            "val_acc": [],
            "val_loss": [],
            "val_step": []
        }

    def validation(self, log=True):
        self.model.eval()
        ground_truth = []
        predicate = []
        with torch.no_grad():
            running_loss = 0.0
            for batch, data in enumerate(tqdm(self.loaders['val'], desc="validating ...")):
                images, labels = [d.to(self.device) for d in data]
                outputs = self.model(images.to(self.device))
                _, predicted = torch.max(outputs, 1)

                loss = self.criterion(outputs, labels)
                ground_truth.extend(labels.cpu())
                predicate.extend(predicted.cpu())
                running_loss += loss.item()
            acc = accuracy_score(ground_truth, predicate)
            running_loss /= (batch+1)
            if log:
                self.log["val_acc"].append(acc)
                self.log["val_loss"].append(running_loss)
                self.log["val_step"].append(self.step)

        print("accuracy score :", acc, "loss :", running_loss)
        return (ground_truth, predicate)

    def train(self):
        print("-- Start Training --")
        for epoch in range(10):
            self.model.train()
            running_loss = 0.0

            # with torch.set_grad_enabled(True):
            for batch, data in enumerate(tqdm(self.loaders['train'], desc="training ...")):
                inputs, labels = [d.to(self.device) for d in data]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(outputs.shape)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                self.step += 1
                _, predicted = torch.max(outputs, 1)
                acc = accuracy_score(labels.cpu(), predicted.cpu())
                self.log["train_acc"].append(acc)
                self.log["train_loss"].append(loss.item())
                self.log["train_step"].append(self.step)

            print(f'[{epoch + 1} epoch, {batch + 1} batches] loss: {running_loss / (batch+1):.3f}')
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