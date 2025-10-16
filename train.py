# train.py

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from utils import *
import pandas as pd
import numpy as np
from os import makedirs
from typing import Union
from dataclasses import dataclass
import warnings
from model import MnistModel  # Import from the new model.py file
warnings.filterwarnings('ignore')


class MnistDataset(data.Dataset):
    """ Custom Dataset for Mnist, now with transforms """
    def __init__(self, df: pd.DataFrame, target: np.array = None, test: bool = False, transform=None) -> None:
        self.df = df
        self.test = test
        self.transform = transform
        if not self.test:
            self.df_targets = target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Union[tuple, torch.Tensor]:
        image = self.df[idx].reshape(28, 28).astype(np.uint8)
        image = Image.fromarray(image, mode='L')

        if self.transform:
            image = self.transform(image)
        else:
            # Default transformation if none provided
            image = transforms.ToTensor()(image)

        if not self.test:
            target = self.df_targets[idx]
            return image, torch.tensor(target, dtype=torch.long)
        else:
            return image


def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Use CrossEntropyLoss since the model now outputs raw logits
    return nn.CrossEntropyLoss()(outputs, targets)


def train_loop_fn(data_loader, model, optimizer, device):
    model.train()
    train_loss = []
    for images, targets in data_loader:
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(x=images)
        loss = loss_fn(outputs, targets)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(train_loss)
    print(f"Loss on Train Data: {avg_loss:.4f}")
    return avg_loss


def eval_loop_fn(data_loader, model, device):
    fin_targets = []
    fin_outputs = []
    model.eval()

    with torch.no_grad():
        for _, (images, targets) in enumerate(data_loader):
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            outputs = model(x=images)
            _, predicted = torch.max(outputs.data, 1)
            fin_targets.append(targets.cpu().numpy())
            fin_outputs.append(predicted.cpu().numpy())

    return np.hstack(fin_outputs), np.hstack(fin_targets)


@timer
def run(args):
    print('Reading Data..')
    dfx = pd.read_csv(args.data_path + 'train.csv', dtype=np.uint8)
    classes = dfx[args.target].nunique()

    print('Data Wrangling..')
    split_idx = int(len(dfx) * 0.9) # Using 90/10 split
    df_train = dfx[:split_idx].reset_index(drop=True)
    df_valid = dfx[split_idx:].reset_index(drop=True)

    train_targets = df_train[args.target].values
    valid_targets = df_valid[args.target].values

    df_train = df_train.drop(args.target, axis=1).values
    df_valid = df_valid.drop(args.target, axis=1).values

    # Data augmentation for the training set
    train_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    ])

    # Only normalization for the validation set
    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print('DataLoader..')
    train_dataset = MnistDataset(df=df_train, target=train_targets, transform=train_transforms)
    valid_dataset = MnistDataset(df=df_valid, target=valid_targets, transform=valid_transforms)
    train_loader = data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MnistModel(classes=classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # CORRECTED LINE: Removed the 'verbose' argument
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    print('Training..')
    best_accuracy = 0
    for epoch in range(args.NUM_EPOCHS):
        print(f'Epoch [{epoch+1}/{args.NUM_EPOCHS}]')
        _ = train_loop_fn(train_loader, model, optimizer, device)
        o, t = eval_loop_fn(valid_loader, model, device)
        accuracy = (o == t).mean() * 100
        print(f'Accuracy on Valid Data: {accuracy:.2f}%')

        scheduler.step(accuracy)

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), args.model_path)
            best_accuracy = accuracy
            print(f'Model saved with accuracy: {best_accuracy:.2f}%')

    print(f'Training Complete. Best Accuracy: {best_accuracy:.2f}%')


if __name__ == "__main__":
    from PIL import Image

    @dataclass
    class Args:
        lr: float = 1e-3
        RANDOM_STATE: int = 42
        NUM_EPOCHS: int = 20
        BATCH_SIZE: int = 128
        target: str = 'label'
        data_path: str = 'data/'
        model_path: str = 'checkpoint/mnist.pt'

        def __post_init__(self):
            makedirs('checkpoint', exist_ok=True)

    arg = Args()
    random_seed(arg.RANDOM_STATE)
    run(args=arg)