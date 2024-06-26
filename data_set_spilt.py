import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 定义数据集路径
dataset_path = 'Emotion6_data/images'
output_path = 'Emotion6_dataset'
categories = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

# 创建输出文件夹
train_path = os.path.join(output_path, 'train')
test_path = os.path.join(output_path, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for category in categories:
    # 创建每个类别的训练集和测试集文件夹
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)

    # 获取每个类别的所有图片路径
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)

    # 划分训练集和测试集
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # 使用tqdm显示进度条
    print(f"正在划分类别: {category}")

    for image in tqdm(train_images, desc=f"复制训练集 {category}"):
        shutil.copy(os.path.join(category_path, image), os.path.join(train_path, category, image))

    for image in tqdm(test_images, desc=f"复制测试集 {category}"):
        shutil.copy(os.path.join(category_path, image), os.path.join(test_path, category, image))

print("数据集划分完成。")
