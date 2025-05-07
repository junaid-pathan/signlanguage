import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set paths
train_path = r'C:\Users\patha\Downloads\sign-language-recognition-project\code\gesture\train'
test_path = r'C:\Users\patha\Downloads\sign-language-recognition-project\code\gesture\test'
model_save_path = r'C:\Users\patha\Downloads\sign-language-recognition-project\code\best_model_dataflair3.h5'

# Load image data
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)

imgs, labels = next(train_batches)

# Plot function with normalized display
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalize for display
        img = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPool2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPool2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D(2, 2),

    Flatten(),
    Dense(64, activation="relu"),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# Compile model
model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Train model
history = model.fit(train_batches, epochs=10, validation_data=test_batches, callbacks=[reduce_lr, early_stop])

# Evaluate
imgs, labels = next(test_batches)
scores = model.evaluate(imgs, labels, verbose=0)
print(f'Loss: {scores[0]:.4f} | Accuracy: {scores[1]*100:.2f}%')

# Save model
model.save(model_save_path)
print(f'\n✅ Model saved to: {model_save_path}')

# Load and evaluate again
model = keras.models.load_model(model_save_path)
scores = model.evaluate(imgs, labels, verbose=0)
print(f'Post-load Evaluation - Loss: {scores[0]:.4f} | Accuracy: {scores[1]*100:.2f}%')
model.summary()

# Predictions
word_dict = {0:'One',1:'Ten',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight',9:'Nine'}
predictions = model.predict(imgs, verbose=0)

print("\n🔮 Predictions:")
for i in predictions:
    print(word_dict[np.argmax(i)], end='   ')

print("\n\n✅ Actual Labels:")
for i in labels:
    print(word_dict[np.argmax(i)], end='   ')

# Show images again
plotImages(imgs)
print(imgs.shape)
