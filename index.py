import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.applications import InceptionV3, ResNet50
import tensorflow as tf

# Define paths for training and testing datasets
train_dataset_path = "E:/BTP_sem8/dataset/vegetable_data_modified/train"
test_dataset_path = "E:/BTP_sem8/dataset/vegetable_data_modified/test"

# Define Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1.0 - 1e-7)
        loss = -y_true * alpha * (1 - y_pred) ** gamma * tf.keras.backend.log(y_pred)
        return tf.keras.backend.mean(loss)
    return loss

# Data augmentation with preprocessing
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = datagen.flow_from_directory(
    train_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Shared Input Layer for Hybrid Model
input_shape = (224, 224, 3)
shared_input = Input(shape=input_shape)

# Load Pre-trained Models (without top layers, using shared input)
base_model1 = ResNet50(weights='imagenet', include_top=False, input_tensor=shared_input)
base_model2 = InceptionV3(weights='imagenet', include_top=False, input_tensor=shared_input)

# Freeze Initial Layers for Feature Extraction
for model in [base_model1, base_model2]:
    for layer in model.layers:
        layer.trainable = False

# Function to Create Base Model Classifiers
def build_model(base_model):
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=shared_input, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=focal_loss(),
                  metrics=['accuracy'])
    return model

# Train Individual Models
trained_models = {
    "resnet": build_model(base_model1),
    "inception": build_model(base_model2)
}

history_dict = {}

for name, model in trained_models.items():
    print(f"Training {name} model...")
    history = model.fit(train_generator, epochs=50, validation_data=test_generator)
    history_dict[name] = history.history
    model.save(f"{name}_model.h5")

# Unfreeze Last 50 Layers for Fine-tuning
for model in [base_model1, base_model2]:
    for layer in model.layers[-50:]:
        layer.trainable = True

# Fine-tune Models
for name, model in trained_models.items():
    print(f"Fine-tuning {name} model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=focal_loss(),
                  metrics=['accuracy'])
    history = model.fit(train_generator, epochs=25, validation_data=test_generator)
    history_dict[f"{name}_fine_tuned"] = history.history
    model.save(f"{name}_fine_tuned_model.h5")

# Create Hybrid Model
x1 = layers.GlobalAveragePooling2D()(base_model1.output)
x2 = layers.GlobalAveragePooling2D()(base_model2.output)

merged = layers.concatenate([x1, x2])
merged = layers.Dense(512, activation='relu')(merged)
merged = layers.Dropout(0.5)(merged)
output_layer = layers.Dense(train_generator.num_classes, activation='softmax')(merged)

hybrid_model = Model(inputs=shared_input, outputs=output_layer)

hybrid_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                     loss=focal_loss(),
                     metrics=['accuracy'])

# Train Hybrid Model
print("Training Hybrid Model...")
hybrid_history = hybrid_model.fit(train_generator, epochs=50, validation_data=test_generator)

hybrid_model.save("final_hybrid_model.h5")
history_dict["Hybrid"] = hybrid_history.history

# Soft Voting Ensemble
models_list = list(trained_models.values()) + [hybrid_model]

def soft_voting_ensemble(models, data_generator):
    y_true = []
    all_preds = []
    steps = len(data_generator)

    for i, (batch_images, batch_labels) in enumerate(data_generator):
        if i >= steps:
            break

        batch_preds = []
        for model in models:
            batch_predictions = model.predict(batch_images, verbose=0)
            batch_preds.append(batch_predictions)

        avg_prediction = np.mean(batch_preds, axis=0)
        all_preds.extend(np.argmax(avg_prediction, axis=1))
        y_true.extend(np.argmax(batch_labels, axis=1))

    return np.array(y_true), np.array(all_preds)

y_true, y_pred = soft_voting_ensemble(models_list, test_generator)

# Validate sizes before computing classification report
if len(y_true) == len(y_pred):
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))
else:
    print(f"Error: y_true({len(y_true)}) and y_pred({len(y_pred)}) mismatch!")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Plot Accuracy and Loss
plt.figure(figsize=(12, 4))

for name, history in history_dict.items():
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label=f"{name} Train Accuracy")
    plt.plot(history['val_accuracy'], label=f"{name} Val Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label=f"{name} Train Loss")
    plt.plot(history['val_loss'], label=f"{name} Val Loss")

plt.subplot(1, 2, 1)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()