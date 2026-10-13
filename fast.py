import cv2
import os

input_folder = "processed/grayscale"
output_folder = "demo/fast"

os.makedirs(output_folder, exist_ok=True)

# FAST detector oluştur
fast = cv2.FastFeatureDetector_create()

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

            # köşeleri bul
            keypoints = fast.detect(gray, None)

            # çiz
            img_with_keypoints = cv2.drawKeypoints(
                img, keypoints, None, color=(255, 0, 0)
            )

            save_path = os.path.join(output_category, filename)
            cv2.imwrite(save_path, img_with_keypoints)

print("FAST tamamlandı!")