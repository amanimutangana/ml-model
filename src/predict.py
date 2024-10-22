import tensorflow as tf
import numpy as np
import cv2
import speech_recognition as sr
import os
import time

def load_model():
    # Load the pre-trained ASL model
    model_path = 'asl_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_frame(frame):
    """Preprocess the webcam frame for prediction."""
    frame = cv2.resize(frame, (64, 64))  # Resize to match model input
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = frame / 255.0  # Normalize pixel values
    return frame

def detect_hands_with_camera():
    """
    Detects hands using the webcam and predicts the ASL gesture for one hand.
    Stops detecting when two hands are present and outputs the text detected up to that point.
    """
    model = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detected_text = ""
    hand_detected = False
    prev_hand_count = 0
    current_prediction = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Detect hands in the frame using OpenCV's hand cascade or a pretrained hand detector.
        hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')  # Example classifier
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hands = hand_cascade.detectMultiScale(gray_frame, 1.1, 4)
        hand_count = len(hands)

        # If two hands are detected, output the detected text and stop prediction
        if hand_count == 2 and prev_hand_count < 2:
            print("Pausing detection. Detected text: ", detected_text)
            cv2.putText(frame, "Pausing detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            prev_hand_count = 2
            cv2.imshow('ASL Detection', frame)
            cv2.waitKey(2000)  # Pause for 2 seconds
            detected_text += " "  # Add space between words
            continue
        elif hand_count == 1:
            prev_hand_count = 1
            hand_detected = True
            for (x, y, w, h) in hands:
                # Draw green rectangle around hand
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Extract the hand region and predict the gesture
                hand_region = frame[y:y + h, x:x + w]
                processed_frame = preprocess_frame(hand_region)
                prediction = model.predict(processed_frame)
                predicted_class = np.argmax(prediction, axis=1)[0]
                current_prediction = chr(predicted_class + ord('A')) if predicted_class >= 10 else str(predicted_class)
                detected_text += current_prediction
                cv2.putText(frame, f'Prediction: {current_prediction}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Draw red rectangle around hand if no valid hand is detected
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Show the frame with the prediction or pause notification
        cv2.imshow('ASL Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Final detected text: ", detected_text)

# def display_black_screen(image_shape, duration=5):
#     """Display a black screen with the same size as the images for a brief moment."""
#     black_image = np.zeros(image_shape, dtype=np.uint8)  # Create a black image with the same shape
#     cv2.imshow('ASL Translation', black_image)
#     cv2.waitKey(duration)  # Display the black screen for the specified duration (in milliseconds)
def display_black_screen(image_shape, duration=500):
    """Display a red screen with the same size as the images for a brief moment."""
    red_image = np.zeros(image_shape, dtype=np.uint8)  # Create an image with the same shape
    red_image[:] = [0, 0, 255]  # Set all pixels to red (BGR format)
    cv2.imshow('ASL Translation', red_image)
    cv2.waitKey(duration)  # Display the red screen for the specified duration (in milliseconds)

# def speech_to_images():
#     """
#     Listens for speech, translates it to text, and displays corresponding ASL images as a slideshow.
#     Includes a red screen when an underscore character is detected.
#     """
#     recognizer = sr.Recognizer()

#     try:
#         with sr.Microphone() as source:
#             print("Listening for speech...")
#             recognizer.adjust_for_ambient_noise(source)
#             audio = recognizer.listen(source)

#             # Recognize speech using Google's speech-to-text API
#             speech_text = recognizer.recognize_google(audio).upper()
#             # Insert underscores between consecutive identical characters
#             processed_text = []
#             for i, char in enumerate(speech_text):
#                 # Always add the current character
#                 processed_text.append(char)
                
#                 # If the next character is the same, add an underscore between them
#                 if i < len(speech_text) - 1 and speech_text[i] == speech_text[i + 1]:
#                     processed_text.append('_')  # Add an underscore between identical letters

#             # Join the list into a new string
#             speech_text = ''.join(processed_text)

#             print(f"Recognized text: {speech_text}")

#             # Convert speech text to ASL images slideshow
#             last_img_shape = None  # Variable to store the shape of the last image
#             for char in speech_text:
#                 if char == "_":
#                     # Display the red screen for 500 milliseconds, using the last image's shape
#                     if last_img_shape is not None:
#                         print("Displaying red screen for underscore")
#                         display_black_screen(last_img_shape, duration=500)  # Show the red screen for 0.5 seconds
#                         cv2.waitKey(1000)  # Show each image for 1 second

#                     else:
#                         print("No previous image to match the red screen size.")
#                 elif char.isalnum():  # Check if char is a letter or digit
#                     image_folder = f"../data/asl_dataset/{char}"  # Folder for the specific letter/digit
#                     if os.path.exists(image_folder):
#                         # Get the first image in the folder
#                         image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#                         if image_files:
#                             image_path = os.path.join(image_folder, image_files[0])  # Use the first image found
#                             print(f"Displaying {image_path}")
#                             img = cv2.imread(image_path)

#                             if img is not None:
#                                 last_img_shape = img.shape  # Store the shape of the last image
#                                 # Display the image
#                                 cv2.imshow(f'ASL Translation: {char}', img)
#                                 cv2.waitKey(1000)  # Show each image for 1 second
#                             else:
#                                 print(f"Error: Could not read {image_path}")
#                         else:
#                             print(f"No image files found for {char}")
#                     else:
#                         print(f"No folder found for {char}")

#     except sr.UnknownValueError:
#         print("Sorry, I could not understand the audio.")
#     except sr.RequestError:
#         print("Error connecting to the speech recognition service.")
#     finally:
#         cv2.destroyAllWindows()

def speech_to_images():
    """
    Listens for speech, translates it to text, and displays corresponding ASL images as a slideshow.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening for speech...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

            # Recognize speech using Google's speech-to-text API
            speech_text = recognizer.recognize_google(audio).upper()

            print(f"Recognized text: {speech_text}")

            # Convert speech text to ASL images slideshow
            for char in speech_text:
                if char.isalnum() or char == "_":  # Check if char is a letter, digit, or underscore
                    image_folder = f"../data/asl_dataset/{char}"  # Folder for the specific letter/digit or underscore
                    if os.path.exists(image_folder):
                        # Get the first image in the folder
                        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if image_files:
                            image_path = os.path.join(image_folder, image_files[0])  # Use the first image found
                            print(f"Displaying {image_path}")
                            img = cv2.imread(image_path)

                            if img is not None:
                                # Display the image
                                cv2.imshow(f'ASL Translation: {char}', img)
                                cv2.waitKey(1000)  # Show each image for 1 second
                            else:
                                print(f"Error: Could not read {image_path}")
                        else:
                            print(f"No image files found for {char}")
                    else:
                        print(f"No folder found for {char}")

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError:
        print("Error connecting to the speech recognition service.")
    finally:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    # Choose one of the functions to run:
    # detect_hands_with_camera()
    speech_to_images()
