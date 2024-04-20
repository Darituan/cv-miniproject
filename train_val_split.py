import os
import random
import shutil


def recreate_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def copy_files(img_dir, ann_dir, img_file, new_img_dir, new_ann_dir):
    # Copy the chosen img files to new folder
    img_src_path = os.path.join(img_dir, img_file)
    img_dest_path = os.path.join(new_img_dir, img_file)
    shutil.copy(img_src_path, img_dest_path)

    # Copy the corresponding ann files to new folder
    ann_file = img_file + ".json"
    ann_src_path = os.path.join(ann_dir, ann_file)
    ann_dest_path = os.path.join(new_ann_dir, ann_file)
    shutil.copy(ann_src_path, ann_dest_path)


def split(img_dir, ann_dir, val_part=0.15):
    train_img_dir = img_dir.replace("/img", "/img_train")
    val_img_dir = img_dir.replace("/img", "/img_val")
    train_ann_dir = ann_dir.replace("/ann", "/ann_train")
    val_ann_dir = ann_dir.replace("/ann", "/ann_val")

    img_files = os.listdir(img_dir)

    val_number = int(len(img_files) * val_part)
    chosen_img_files = random.sample(img_files, val_number)

    # Remove training and validation folders with images and annotations and create them again
    recreate_dir(val_img_dir)
    recreate_dir(train_img_dir)
    recreate_dir(val_ann_dir)
    recreate_dir(train_ann_dir)

    for img_file in chosen_img_files:
        copy_files(img_dir, ann_dir, img_file, val_img_dir, val_ann_dir)

    for img_file in img_files:
        if img_file not in chosen_img_files:
            copy_files(img_dir, ann_dir, img_file, train_img_dir, train_ann_dir)

    print(f"{val_number} files were randomly chosen from img and copied to img_val, along with their corresponding "
          f"ann files copied to ann_val.")
    print(f"The remaining files were copied to img_train and ann_train.")


if __name__ == "__main__":
    split("dataset/parts/File1/img", "dataset/parts/File1/ann")
    split("dataset/damage/File1/img", "dataset/damage/File1/ann")
