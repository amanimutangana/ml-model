import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import os

def load_data_for_class(directory, image_size=(64, 64)):
    """
    Loads the data from a specific directory for a single class (letter/digit).
    
    Args:
    - directory (str): The path to the directory containing the images.
    - image_size (tuple): The size to which the images will be resized.
    
    Returns:
    - np.array: Array of images loaded and resized.
    """
    image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.lower().endswith('.jpeg')]
    images = []
    
    for img_path in image_paths:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        images.append(img_array)
    
    return np.array(images)

def test_model_on_class(model, class_dir, class_label, image_size=(64, 64)):
    """
    Tests the model on a single class (letter/digit) and returns the accuracy.
    
    Args:
    - model: The loaded Keras model.
    - class_dir (str): Directory containing images of a single class (letter/digit).
    - class_label (int): The label corresponding to the class.
    
    Returns:
    - float: Accuracy of the model on the given class.
    """
    images = load_data_for_class(class_dir, image_size)
    if len(images) == 0:
        print(f"No images found in {class_dir}")
        return None

    labels = np.full(len(images), class_label)
    
    # Predict class probabilities for all images
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy for this class
    accuracy = accuracy_score(labels, predicted_classes)
    return accuracy

def test_model():
    # Load the trained model
    model_path = '../src/asl_model.h5'  # Adjust if the model is located elsewhere
    model = tf.keras.models.load_model(model_path)
    
    data_dir = '../data/validate'  # Directory containing the validation data for all classes
    class_labels = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]  # 0-9 and A-Z
    
    # Dictionary to store accuracy for each class
    class_accuracies = {}

    # Test the model on each class
    for idx, label in enumerate(class_labels):
        # Handle lowercase folder names by checking both upper and lower case
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            class_dir = os.path.join(data_dir, label.lower())  # Fallback to lowercase folder if uppercase not found
        
        accuracy = test_model_on_class(model, class_dir, idx)
        if accuracy is not None:
            class_accuracies[label] = accuracy
            print(f"Accuracy for {label}: {accuracy * 100:.2f}%")
    
    # Print the summary of results
    print("\n=== Class-wise Accuracy Summary ===")
    for label, accuracy in class_accuracies.items():
        print(f"Class {label}: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_model()
