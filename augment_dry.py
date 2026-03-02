import os
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

target_count = 650   # match normal class size

root = Path("clean_dataset/train/dry")
current_images = list(root.glob("*"))

print("Current dry images:", len(current_images))

datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.7, 1.3],
    zoom_range=0.15,
    horizontal_flip=True
)

i = 0
while len(list(root.glob("*"))) < target_count:
    img_path = current_images[i % len(current_images)]
    img = Image.open(img_path).convert("RGB")
    x = np.array(img)
    x = x.reshape((1,) + x.shape)

    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=root,
                              save_prefix="aug",
                              save_format="jpg"):
        break

    i += 1

print("Done — new dry count:", len(list(root.glob('*'))))
