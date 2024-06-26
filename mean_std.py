import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_mean_and_std(root_folder):
    image_paths = []

    # 遍历根文件夹下的所有子文件夹，并获取所有图像文件的路径
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(subdir, file))

    if not image_paths:
        raise ValueError("No images found in the specified folder.")

    mean = np.zeros(3)
    std = np.zeros(3)
    num_pixels = 0

    for image_path in tqdm(image_paths, desc="Processing images"):
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image) / 255.0  # Normalize to [0, 1]

        mean += image_np.mean(axis=(0, 1))
        std += image_np.std(axis=(0, 1))
        num_pixels += 1

    mean /= num_pixels
    std /= num_pixels

    return mean, std


# 使用方法
root_folder = 'cat_dog_data/training_set'  # 替换为你的根文件夹路径
mean, std = compute_mean_and_std(root_folder)
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

