import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, regularizers

# ------------------------
# Paths & settings
# ------------------------
data_dir = "clean_dataset"  # Make sure this exists with train/valid folders
img_size = (224, 224)
batch_size = 8  # Smaller batch size helps with small datasets

# ------------------------
# Data Generators
# ------------------------
# Train generator with aggressive augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation generator - only rescale
valid_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    os.path.join(data_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    os.path.join(data_dir, "valid"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# ------------------------
# Base model - MobileNetV2
# ------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all except last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

# ------------------------
# Classification Head
# ------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),  # dropout after pooling
    layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.002)
    ),
    layers.Dropout(0.5),
    layers.Dense(3, activation="softmax")
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------
# Callbacks
# ------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "skin_type_model_v2.h5",
    monitor="val_loss",
    save_best_only=True
)

# ------------------------
# Train the model
# ------------------------
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

print("\nTraining complete — best model saved as skin_type_model_v2.h5")