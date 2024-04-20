import torch
import torchvision
from dataset import CarDataset
from torch.utils.data import DataLoader
from metadata import retrieve_meta_data


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_ann_dir,
        val_dir,
        val_ann_dir,
        classes,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = CarDataset(train_dir, train_ann_dir, classes, train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarDataset(val_dir, val_ann_dir, classes, val_transform)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
