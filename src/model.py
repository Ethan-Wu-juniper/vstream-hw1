import torch
import torch.nn as nn
from config import Variables

class CNN(torch.nn.Module):
    def __init__(self, num_classes=Variables.NUM_CLASS):
        super(CNN, self).__init__()
        self.stage1 = self._make_stage(3, 64, num_blocks=1, max_pooling=True)
        self.stage2 = self._make_stage(64, 128, num_blocks=1, max_pooling=True)
        self.stage3 = self._make_stage(128, 64, num_blocks=1, max_pooling=True)
        self.stage4 = self._make_stage(64, 64, num_blocks=1, max_pooling=True)

        self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(64*2*2, num_classes)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()

    def _make_stage(self, in_dim, out_dim, num_blocks, max_pooling):
        stage = nn.Sequential()
        for i in range(num_blocks):
            stage.add_module(f"conv{i+1}", nn.Conv2d(
                in_channels=in_dim if i==0 else out_dim,
                out_channels=out_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            stage.add_module(f"relu{i+1}", nn.ReLU())
        if max_pooling:
            stage.add_module("maxpool", nn.MaxPool2d(kernel_size=3, stride=3))

        return stage

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        logits = self.classifier(x)

        return logits