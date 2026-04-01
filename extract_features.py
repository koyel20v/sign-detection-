import cv2
import os
import numpy as np

data = []
labels = []

dataset_path = "dataset"

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize image
        img = cv2.resize(img, (64, 64))

        # Convert to grayscale (optional)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Flatten image
        flat = gray.flatten()

        data.append(flat)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

np.save("data.npy", data)
np.save("labels.npy", labels)

print("✅ Done without MediaPipe!")