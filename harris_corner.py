import cv2
import os
import numpy as np

input_folder = "processed/grayscale"
output_folder = "demo/harris"

os.makedirs(output_folder, exist_ok=True)

for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)

    if os.path.isdir(category_path):
        output_category = os.path.join(output_folder, category)
        os.makedirs(output_category, exist_ok=True)

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)

            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)

            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)

            img[dst > 0.01 * dst.max()] = [0, 0, 255]

            save_path = os.path.join(output_category, filename)
            cv2.imwrite(save_path, img)

print("Harris corner detection tamamlandı!")