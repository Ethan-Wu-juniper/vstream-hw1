from torch.utils.data import DataLoader

from train import ImageClassifier
from dataset import load_dataset
from model import CNN
from config import Variables

datasets = {
    split: load_dataset(split) for split in Variables.SPLIT
}
dataloaders = {
    split: DataLoader(datasets[split], batch_size=Variables.BATCH_SIZE, shuffle=False)
    for split in datasets
}
model_test = CNN()
clf_test = ImageClassifier.from_state(model_test, dataloaders, "HW1_0816004.pt")
out_df = clf_test.test()
out_df.to_csv("HW1_0816004.csv", index=False)