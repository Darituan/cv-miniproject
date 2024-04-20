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
