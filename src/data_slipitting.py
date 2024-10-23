import os
import shutil
import random

def split_dataset(dataset_dir='../data/asl_dataset', training_dir='../data/training', validation_dir='../data/validate', split_ratio=0.8):
    """
    Splits the dataset into training and validation sets.

    Args:
    - dataset_dir (str): The directory containing the dataset (folders of images).
    - training_dir (str): The directory to store training data.
    - validation_dir (str): The directory to store validation data.
    - split_ratio (float): The ratio of training data. Default is 0.8 (80% training, 20% validation).
    """
    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        # Create corresponding label folders in training and validation directories
        os.makedirs(os.path.join(training_dir, label), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, label), exist_ok=True)
        
        images = os.listdir(label_path)
        random.shuffle(images)
        
        split_point = int(len(images) * split_ratio)
        training_images = images[:split_point]
        validation_images = images[split_point:]
        
        # Move training images
        for img in training_images:
            shutil.copy(os.path.join(label_path, img), os.path.join(training_dir, label, img))
        
        # Move validation images
        for img in validation_images:
            shutil.copy(os.path.join(label_path, img), os.path.join(validation_dir, label, img))

if __name__ == "__main__":
    split_dataset()
