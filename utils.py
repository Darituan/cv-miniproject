import random

import torch
import torchvision
import matplotlib.pyplot as plt
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


def check_accuracy(loader, model, device="cuda"):
    accuracy = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid((model(x)))
            preds = (preds > 0.5).float()
            num_correct = (preds == y).sum()
            num_pixels = torch.numel(preds)
            accuracy += num_correct/num_pixels
            dice_scores = []
            for i in range(len(preds)):
                pred = preds[i]
                current_y = y[i]
                dice_scores.append((2 * (pred * current_y).sum()) / ((pred + current_y).sum() + 1e-8))
            dice_score += torch.mean(torch.tensor(dice_scores))
        accuracy /= len(loader)
    print(
        f"Accuracy: {accuracy*100:.2f}%"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def mask_to_image(mask, classes, associated_colors):
    pass


# modify for multiple classes
def save_predictions_as_imgs(
        loader, model, classes, associated_colors, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}/correct_{idx}.png")

    model.train()


def visualize_image_and_masks(image, output, target):
    col = 3
    row = 1
    fig, ax = plt.subplots(row,col)
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(output)
    ax[1].set_title("Output")
    ax[2].imshow(target)
    ax[2].set_title("Target")
    plt.show()


def visualize(loader, model, device="cuda", num_examples=5):
    model.eval()
    with torch.no_grad():
        xs, ys = next(iter(loader))
        xs = xs.to(device)
        ys = ys.to(device)
        preds = torch.sigmoid((model(xs)))
        preds = (preds > 0.5).float()
        indices = random.sample(range(1, len(xs)), num_examples)
        for i in indices:
            x = xs[i]
            y = ys[i]
            pred = preds[i]
            image = x.permute(1, 2, 0).cpu()
            target = y.argmax(dim=0).cpu()
            output = pred.argmax(dim=0).cpu()
            visualize_image_and_masks(image, output, target)
            plt.show()
    model.train()


