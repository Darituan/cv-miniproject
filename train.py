import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from metadata import retrieve_meta_data
from model import UNet

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_PARTS_IMG_DIR = "dataset/parts/File1/img_train"
VAL_PARTS_IMG_DIR = "dataset/parts/File1/img_val"
TRAIN_PARTS_ANN_DIR = "dataset/parts/File1/ann_train"
VAL_PARTS_ANN_DIR = "dataset/parts/File1/ann_val"
PARTS_META_PATH = "dataset/parts/meta.json"

TRAIN_DAMAGE_IMG_DIR = "dataset/damage/File1/img_train"
VAL_DAMAGE_IMG_DIR = "dataset/damage/File1/img_val"
TRAIN_DAMAGE_ANN_DIR = "dataset/damage/File1/ann_train"
VAL_DAMAGE_ANN_DIR = "dataset/damage/File1/ann_val"
DAMAGE_META_PATH = "dataset/damage/meta.json"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def make_train_transform():
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(transpose_mask=True),
        ],
    )
    return transform


def make_val_transform():
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(transpose_mask=True),
        ],
    )
    return transform


def train_on(train_dir, train_ann_dir, val_dir, val_ann_dir, meta_path, checkpoint_filename="my_checkpoint.pth.tar"):
    train_transform = make_train_transform()
    val_transforms = make_val_transform()

    classes, class_titles, associated_colors = retrieve_meta_data(meta_path)
    out_channels = len(classes) + 1

    model = UNet(in_channels=3, out_channels=out_channels).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        train_dir,
        train_ann_dir,
        val_dir,
        val_ann_dir,
        classes,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(checkpoint_filename), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, checkpoint_filename)
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, device=DEVICE)


def main():
    train_on(TRAIN_PARTS_IMG_DIR, TRAIN_PARTS_ANN_DIR, VAL_PARTS_IMG_DIR, VAL_PARTS_ANN_DIR, PARTS_META_PATH)


if __name__ == "__main__":
    main()
