import tensorflow as tf
from model import build_asl_model
import os

def train_model():
    # Load the training and validation datasets
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '../data/training',  # Adjusted path
        image_size=(64, 64),
        batch_size=32,
        label_mode='categorical'
    )
    
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '../data/validate',  # Adjusted path
        image_size=(64, 64),
        batch_size=32,
        label_mode='categorical'
    )
    
    # Normalize pixel values between 0 and 1
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
    
    # Build the model
    model = build_asl_model()
    
    # Train the model
    model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
    
    # Save the trained model in src directory
    model.save('asl_model.h5')
    print("Model saved as asl_model.h5")

if __name__ == "__main__":
    train_model()
