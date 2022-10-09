import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import *
from tqdm.auto import tqdm
from uuid import uuid1
import os
import matplotlib.pyplot as plt
import pandas as pd

from dataset import load_dataset
from model import CNN
from config import Variables

class ImageClassifier:
    def __init__(self, model, dataloaders, validate=True):
        self.id = str(uuid1())[:5]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(model.parameters(), lr=Variables.LR)
        self.loaders = dataloaders
        self.validate = validate
        self.log_step = 1
        self.step = 0
        self.min_loss = float("inf")
        self.state_dict = []
        self.log = {
            "train_acc": [],
            "train_loss": [],
            "train_step": [],
            "val_acc": [],
            "val_loss": [],
            "val_step": []
        }

    @classmethod
    def from_state(cls, model, dataloaders, state_dict):
        model.load_state_dict(torch.load(state_dict))
        return cls(model, dataloaders)

    @staticmethod
    def mkdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def plot(self, plot_type, step=10, show=True, dump=True):
        log = self.log
        train_step = list(filter(lambda data: data[1]%step==0 or data[0]==len(log['train_step'])-1, enumerate(log['train_step'])))
        train_step = [i[0] for i in train_step]
        train_y = [log[f'train_{plot_type}'][i] for i in train_step]
        plt.plot(train_step, train_y, color='blue', label=f"training {plot_type}")
        plt.plot(log['val_step'], log[f'val_{plot_type}'], color='yellow', label=f"validation {plot_type}")
        limit = 1 if plot_type=="acc" else 3
        plt.ylim(0, limit)
        plt.legend()

        if dump:
            path = "plot"
            self.mkdir(path)
            plt.savefig(f'{path}{self.id}_{plot_type}.png')
        if show:
            plt.show()

    def save_state(self):
        print("-- saving dict --")
        self.state_dict.append((self.min_loss, self.model.state_dict()))
        self.state_dict = sorted(self.state_dict, key=lambda x: x[0])[:3]

    def dump_state(self):
        path = "state_dict"
        self.mkdir(path)
        for loss, state in self.state_dict:
            torch.save(state, f"{path}/{self.id}_loss{loss:.3f}.pt")

    def test(self):
        self.model.eval()
        predicate = []
        with torch.no_grad():
            running_loss = 0.0
            for batch, data in enumerate(tqdm(self.loaders['test'], desc="testing ...")):
                images, path = data
                outputs = self.model(images.to(self.device))
                _, predicted = torch.max(outputs, 1)
                predicate.extend(list(zip(path, predicted.cpu().tolist())))
        df = pd.DataFrame(predicate, columns=["name", "label"]).sort_values(by="name", key=lambda path: path.map(lambda x: int(x.replace(".jpg", ""))))

        return df


    def validation(self, log=True):
        self.model.eval()
        ground_truth = []
        predicate = []
        with torch.no_grad():
            running_loss = 0.0
            for batch, data in enumerate(tqdm(self.loaders['val'], desc="validating ...")):
                images, labels = [d.to(self.device) for d in data]
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                # print("outputs :", outputs)
                # print("predicted :", predicted)

                loss = self.criterion(outputs, labels)
                ground_truth.extend(labels.cpu().tolist())
                predicate.extend(predicted.cpu().tolist())
                running_loss += loss.item()
            acc = accuracy_score(ground_truth, predicate)
            running_loss /= (batch+1)
            if log:
                self.log["val_acc"].append(acc)
                self.log["val_loss"].append(running_loss)
                self.log["val_step"].append(self.step)

        if self.min_loss > running_loss:
            self.min_loss = running_loss
            self.save_state()

        print("accuracy score :", acc, "loss :", running_loss)
        return (ground_truth, predicate)

    def train(self, num_epoch=10, num_patient=3):
        print("-- Start Training --")
        for epoch in range(num_epoch):
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
    model = CNN()
    datasets = {
        split: load_dataset(split) for split in Variables.SPLIT
    }
    dataloaders = {
        split: DataLoader(datasets[split], batch_size=Variables.BATCH_SIZE, shuffle=False)
        for split in datasets
    }

    clf = ImageClassifier(model, dataloaders)

    clf.train(100)
    clf.plot("loss", step=10)
    print(clf.get_num_params())
    clf.dump_state()
