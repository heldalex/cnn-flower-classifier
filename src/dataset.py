import os
import shutil
import random
import sys
import tarfile
import urllib.request

DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
OUTPUT_PATH = "17flowers.tgz"
FLOWERS_DIR = '17flowers'
FLOWERS_DATASET_DIR = f'{FLOWERS_DIR}/jpg'
TXT_FILE = 'files.txt'
PROCESSED_DATASET_DIR = 'dataset'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'


def read_files_txt(file_path):
    """
    Reads and parses the files.txt file.
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    dataset = {}
    for i, line in enumerate(lines):
        class_id = i // 80 + 1
        dataset.setdefault(class_id, []).append(line.strip())

    return dataset


def organize_dataset(dataset):
    """
    Splits the dataset into train, validation, and test sets and organizes files.
    """

    random.seed(42)
    for class_id, images in dataset.items():
        train = random.sample(images, int(0.5 * len(images)))
        set_minus_train = list(set(images) - set(train))
        val = random.sample(set_minus_train, int(0.5 * len(set_minus_train)))
        test = list(set(set_minus_train) - set(val))

        for split, split_images in [(TRAIN, train), (VAL, val), (TEST, test)]:
            split_dir = os.path.join(
                PROCESSED_DATASET_DIR, split, str(class_id))
            os.makedirs(split_dir, exist_ok=True)
            for image in split_images:
                shutil.copy(os.path.join(
                    FLOWERS_DATASET_DIR, image), split_dir)


if not os.path.exists(OUTPUT_PATH):
    print("Downloading Flowers dataset...")
    try:
        urllib.request.urlretrieve(DATASET_URL, OUTPUT_PATH)
        print("Dataset Flowers downloaded.\n")
    except Exception as e:
        print(f"Failed to download Flowers dataset: {e}")

if not os.path.exists(FLOWERS_DIR):
    print("Extracting Flowers dataset...")
    try:
        with tarfile.open(OUTPUT_PATH, "r:gz") as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(path=FLOWERS_DIR, filter='tar')
            else:
                tar.extractall(path=FLOWERS_DIR)
        print(f"Dataset extracted to {FLOWERS_DIR}\n")
    except Exception as e:
        print(f"Failed to extract dataset Flowers: {e}")

try:
    os.remove(OUTPUT_PATH)
except Exception as e:
    print(f"Failed to delete archive: {e}")

if not os.path.exists(PROCESSED_DATASET_DIR):
    print("Preparing Flowers dataset...")
    try:
        dataset = read_files_txt(os.path.join(FLOWERS_DATASET_DIR, TXT_FILE))
        organize_dataset(dataset)
        print("Dataset preparation successful.\n")
    except Exception as e:
        print(f"Failed to prepare dataset: {e}")

try:
    shutil.rmtree(FLOWERS_DIR)
except Exception as e:
    print(f"Failed to clean up raw dataset: {e}")
