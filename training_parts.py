import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def load_path(path):
    """
    Load X-ray dataset and return a list of dictionaries containing image paths and labels.
    """
    dataset = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for body in os.listdir(folder_path):
                body_path = os.path.join(folder_path, body)
                for patient_id in os.listdir(body_path):
                    patient_path = os.path.join(body_path, patient_id)
                    for label_folder in os.listdir(patient_path):
                        label = 'fractured' if label_folder.endswith('positive') else 'normal'
                        label_path = os.path.join(patient_path, label_folder)
                        for img in os.listdir(label_path):
                            img_path = os.path.join(label_path, img)
                            dataset.append({
                                'label': label,
                                'image_path': img_path
                            })
    return dataset

# Set up paths and load data
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(THIS_FOLDER, 'Dataset')
data = load_path(image_dir)

# Prepare data for training
labels = [row['label'] for row in data]
filepaths = [row['image_path'] for row in data]
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

# Split the data into training and test sets
train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

# Data generators for training, validation, and testing
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# Creating data generators
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Load the pre-trained ResNet50 model
pretrained_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Freeze the base model
pretrained_model.trainable = False

# Building the model
inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)  # Change to 2 classes
model = tf.keras.Model(inputs, outputs)
print(model.summary())

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks])

# Save the trained model
model.save(os.path.join(THIS_FOLDER, "weights/ResNet50_BodyParts.h5"))

# Evaluate the model
results = model.evaluate(test_images, verbose=0)
print(results)
print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
