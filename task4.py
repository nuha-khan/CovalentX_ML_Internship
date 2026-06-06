import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

dataset_path = "leapGestRecog"

X = []
y = []

gesture_labels = {
    "01_palm" : 0, "02_l" : 1, "03_fist" : 2, "04_fist_moved" : 3, "05_thumb" : 4, 
    "06_index" : 5, "07_ok" : 6, "08_palm_moved" : 7, "09_c" : 8, "10_down" : 9
}

# Loading and preprocessing the dataset
for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path,person_folder)
    for gesture_folder in os.listdir(person_path):
        gesture_path = os.path.join(person_path,gesture_folder)
        for image_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path,image_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img,(64,64))               # Resize all images to a fixed size
            img = img/255.0                             # Normalize pixel values between 0 and 1
            X.append(img)
            y.append(gesture_labels[gesture_folder])

# Convert lists into NumPy arrays for model training
X = np.array(X)
y = np.array(y)

# Split dataset into training and testing sets while preserving class distribution
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense

model = Sequential()                            # Building the CNN model
model.add(Input(shape=(64,64,3)))
model.add(Conv2D(32,(3,3),activation="relu"))   # First convolutional layer to detect basic features such as edges and shapes
model.add(MaxPooling2D(pool_size=(2,2)))        # Reduce feature map size while retaining important information
model.add(Conv2D(64,(3,3),activation="relu"))   # Second convolutional layer to learn more complex patterns
model.add(MaxPooling2D(pool_size=(2,2)))        # Further downsampling of feature maps
model.add(Flatten())                            # Convert multidimensional feature maps into a 1D vector
model.add(Dense(128,activation="relu"))         # Fully connected layer for learning gesture-specific patterns
model.add(Dense(10,activation="softmax"))       # Output layer with 10 gesture classes
# model.summary()

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])     # Configure the model for training
history = model.fit(X_train,y_train,epochs=3,validation_data=(X_test,y_test),verbose=0)         # Train the CNN model on the training dataset

# Display training and validation accuracy
print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# Save the trained model for future use
model.save("hand_gesture_model.keras")
print("Model saved successfully!")

# Real-world image prediction
class_names = {0: "Palm",1: "L",2: "Fist",3: "Fist Moved",4: "Thumb",5: "Index",6: "OK",
               7: "Palm Moved",8: "C",9: "Down"
}

user_image = input("Enter image path: ")

img = cv2.imread(user_image)

if img is None:
    print("Invalid image path!")
else:
    img = cv2.resize(img, (64, 64))             # Same preprocessing as training
    img = img / 255.0
    img = img.reshape(1, 64, 64, 3)             # CNN expects: (batch_size, height, width, channels)
    prediction = model.predict(img, verbose=0)  # Predict
    predicted_class = np.argmax(prediction)     # Get class with highest probability
    confidence = np.max(prediction)             # Get confidence score

    print(f"\nPredicted Class ID: {predicted_class}")
    print(f"Predicted Gesture: {class_names[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")