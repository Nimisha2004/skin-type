import cv2
import os
from pathlib import Path

cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

input_root = Path("dataset")
output_root = Path("clean_dataset")

output_root.mkdir(exist_ok=True)

for split in ["train", "test", "valid"]:
    for label in ["oily", "normal", "dry"]:
        input_folder = input_root / split / label
        output_folder = output_root / split / label

        output_folder.mkdir(parents=True, exist_ok=True)

        for img_name in os.listdir(input_folder):
            img_path = str(input_folder / img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(60, 60)
            )

            # if no face → skip
            if len(faces) == 0:
                continue

            # take the largest detected face (just one)
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_img = img[y:y+h, x:x+w]

            save_path = str(output_folder / img_name)
            cv2.imwrite(save_path, face_img)

print("Done — cropped faces saved in clean_dataset/")
