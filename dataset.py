import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from skimage.draw import polygon
import numpy as np


class CarDataset(Dataset):
    def __init__(self, image_dir, ann_dir, classes, transform=None):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.classes = classes
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def get_mask(self, annfile):
        img_height, img_width = annfile["size"]["height"], annfile["size"]["width"]
        mask = torch.zeros((img_height, img_width), dtype=torch.long)
        mask_numpy = mask.numpy()
        class_ids = []
        for object_ in annfile["objects"]:
            class_id = self.classes.index(object_["classId"])
            class_ids.append(class_id)
            polygon_ = []
            for i in object_["points"]["exterior"]:
                polygon_.append(i)
            polygon_ = np.asarray(polygon_)
            x_, y_ = polygon(polygon_[:, 0], polygon_[:, 1], (annfile["size"]["width"], annfile["size"]["height"]))
            mask_numpy[y_, x_] = class_id + 1

        masks = [(mask_numpy == class_id) for class_id in range(len(self.classes) + 1)]
        masks_numpy = np.stack(masks, axis=-1).astype('int')
        return masks_numpy

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        ann_path = os.path.join(self.ann_dir, self.images[item].replace(".png", ".png.json"))
        img = np.array(Image.open(img_path).convert("RGB"))
        ann = json.load(open(ann_path, "r"))

        annotated_mask = self.get_mask(ann)

        if self.transform is not None:
            augmentations = self.transform(image=img, mask=annotated_mask)
            img = augmentations["image"]
            annotated_mask = augmentations["mask"]

        return img, annotated_mask
