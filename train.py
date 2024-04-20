import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from metadata import retrieve_meta_data
from model import UNet
# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_imgs,
# )


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
PIN_MEMORY = True
LOAD_MODEL = False

PARTS_IMG_DIR = "dataset/parts/File1/img"
PARTS_ANN_DIR = "dataset/parts/File1/ann"
PARTS_META_PATH = "dataset/parts/meta.json"

DAMAGE_IMG_DIR = "dataset/damage/File1/img"
DAMAGE_ANN_DIR = "dataset/damage/File1/ann"
DAMAGE_META_PATH = "dataset/damage/meta.json"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

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
            ToTensorV2(),
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
            ToTensorV2(),
        ],
    )
    return transform


def train_on(img_dir, ann_dir, meta_path):
    train_transform = make_train_transform()
    val_transforms = make_val_transform()

    classes, class_titles, associated_colors = retrieve_meta_data(meta_path)
    out_channels = len(classes) + 1

    model = UNet(in_channels=3, out_channels=out_channels).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        img_dir,
        ann_dir,
        classes,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        # check accuracy
        # print some examples to a folder


def main():
    train_on(PARTS_IMG_DIR, PARTS_ANN_DIR, PARTS_META_PATH)


if __name__ == "__main__":
    main()
