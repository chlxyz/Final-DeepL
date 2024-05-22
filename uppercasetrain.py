import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# Define the list of labels
labels_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Load images and labels
path = r'D:\winback\attempt12\dataset-upper'
images = []
labels = []

dir_list = os.listdir(path)
for i in dir_list:
    dir = os.path.join(path, i)
    file_list = os.listdir(dir)
    for j in file_list:
        files = os.path.join(dir, j)
        img = cv2.imread(files)
        img = cv2.resize(img, (32, 32))
        img = np.array(img, dtype=np.float32)
        img = img / 255
        images.append(img)
        labels.append(i)

# Total number of images in the dataset
total_images = len(images)
print(f"Total number of images in the dataset: {total_images}")

# Data shuffling and preprocessing
X = np.array(images)
y = np.array(labels)
le = LabelEncoder()
y = le.fit_transform(y)
X_sh, y_sh = shuffle(X, y, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(X_sh)

# Build the model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=len(labels_list), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define an early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

X_train, X_val, y_train, y_val = train_test_split(X_sh, y_sh, test_size=0.2, random_state=42)

# Train the model with the early stopping callback
history = model.fit(datagen.flow(X_train, y_train, batch_size=50),
                    validation_data=(X_val, y_val),
                    epochs=100, callbacks=[early_stopping])

# Save the trained model
model.save('uppercasemodel.h5')

# Access the training history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Create a DataFrame to store the training history
history_df = pd.DataFrame({
    'Epoch': range(1, len(training_loss) + 1),
    'Training Loss': training_loss,
    'Validation Loss': validation_loss,
    'Training Accuracy': training_accuracy,
    'Validation Accuracy': validation_accuracy
})

# Display the table
print(history_df)

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_df['Epoch'], history_df['Training Loss'], label='Training Loss')
plt.plot(history_df['Epoch'], history_df['Validation Loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_df['Epoch'], history_df['Training Accuracy'], label='Training Accuracy')
plt.plot(history_df['Epoch'], history_df['Validation Accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
