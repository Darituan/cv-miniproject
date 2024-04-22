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


# modify for multiple classes
def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
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
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}correct_{idx}.png")

    model.train()
