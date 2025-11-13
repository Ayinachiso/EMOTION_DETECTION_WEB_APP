# ------------------------------------------------------------
# model.py - CNN Model Training Script for Emotion Detection
# ------------------------------------------------------------
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# ------------------------------------------------------------
# 2. Data Setup
# ------------------------------------------------------------
train_dir = 'train'
test_dir = 'test'

# Data augmentation for training, simple rescale for testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

print("✅ Class indices:", train_generator.class_indices)

# ------------------------------------------------------------
# 3. Model Architecture
# ------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------------------------------------------------
# 4. Callbacks
# ------------------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("models", exist_ok=True)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='checkpoints/best_face_emotionModel.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ------------------------------------------------------------
# 5. Training
# ------------------------------------------------------------
try:
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=test_generator,
        callbacks=[early_stop, checkpoint]
    )

except KeyboardInterrupt:
    print("\n🛑 Training interrupted manually. Saving current model...")

finally:
    # Always save final model
    model.save("models/face_emotionModel.h5")
    print("✅ Model saved successfully as models/face_emotionModel.h5")

    # Plot accuracy progress
    if 'history' in locals():
        plt.plot(history.history['accuracy'], label='Train acc')
        plt.plot(history.history['val_accuracy'], label='Val acc')
        plt.title("Training Progress")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()