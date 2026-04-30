import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import sys

def compute_mean_std(txt_path, root_path):
    if not os.path.exists(txt_path):
        print(f"❌ 错误: 找不到文件 {txt_path}")
        return None, None

    with open(txt_path, 'r') as f:
        img_names = [line.strip() for line in f.readlines() if line.strip()]

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    found_count = 0

    print(f"🔍 正在检查路径并处理 {len(img_names)} 张图片...")

    for name in tqdm(img_names):
        # 核心调试点：确保路径拼接结果是你电脑上真实的路径
        img_path = os.path.join(root_path, name)
        
        if not os.path.exists(img_path):
            # 第一次失败时提醒你正确的路径应该是什么
            if found_count == 0 and pixel_count == 0:
                print(f"\n⚠️ 找不到文件，请检查！\n尝试路径: {os.path.abspath(img_path)}")
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img).astype(np.float64)
            
            channel_sum += np.sum(img_array, axis=(0, 1))
            channel_sq_sum += np.sum(np.square(img_array), axis=(0, 1))
            pixel_count += img_array.shape[0] * img_array.shape[1]
            found_count += 1
        except Exception as e:
            print(f"读取失败 {img_path}: {e}")

    if pixel_count == 0:
        print("\n❌ 失败: 没有任何图片被成功读取！请检查 train.txt 里的路径和 ROOT_DIR。")
        return None, None

    mean = channel_sum / pixel_count
    std = np.sqrt(np.maximum(channel_sq_sum / pixel_count - np.square(mean), 0))

    print(f"\n✅ 成功处理了 {found_count} 张图片")
    return mean, std

if __name__ == "__main__":
    # --- ！！！请务必确认这两个路径 ！！！---
    # 如果你在 FusNet 目录下运行，ROOT_DIR 写 "./"
    # 如果图片在子目录里，比如 FusNet/data/train/，记得对应好
    TRAIN_TXT_FILE = "train.txt" 
    ROOT_DIR = "./" 
    # --------------------------------------

    mean, std = compute_mean_std(TRAIN_TXT_FILE, ROOT_DIR)

if mean is not None:
        # 将 numpy 数组转为标准 Python 列表，并保留 4 位小数
        mean_list = [round(float(x), 4) for x in mean]
        std_list = [round(float(x), 4) for x in std]

        print("\n" + "="*40)
        print("计算完成（针对 0-255 像素）：")
        print(f"Mean (R, G, B): {mean_list}")
        print(f"Std  (R, G, B): {std_list}")
        print("="*40)
        
        print(f"\n直接复制到 Albumentations:")
        print(f"A.Normalize(mean={mean_list}, std={std_list}, max_pixel_value=255.0)")