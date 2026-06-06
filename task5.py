import os
import cv2
import numpy as np

dataset = "Dataset"

food_labels = {
    "burger" : 0, "butter_naan" : 1, "chai" : 2, "chapati" : 3, "chole_bhature" : 4, "dal_makhani" : 5,
    "dhokla" : 6, "fried_rice" : 7, "idli" : 8, "jalebi" : 9, "kaathi_rolls" : 10, "kadai_paneer" : 11,
    "kulfi" : 12, "masala_dosa" : 13, "momos" : 14, "paani_puri" : 15, "pakode" : 16, "pav_bhaji" : 17,
    "pizza" : 18, "samosa" : 19
}

# Loading and preprocessing the training dataset
X_train = []
y_train = []
train_path = os.path.join(dataset, "train") 

for food_item in os.listdir(train_path):
    food_item_path = os.path.join(train_path, food_item)

    for image_name in os.listdir(food_item_path):
        img_path = os.path.join(food_item_path, image_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        img = img / 255.0

        X_train.append(img)
        y_train.append(food_labels[food_item])

# Loading and preprocessing the validation dataset
X_val = []
y_val = []
val_path = os.path.join(dataset, "val")

for food_item in os.listdir(val_path):
    food_item_path = os.path.join(val_path, food_item)

    for image_name in os.listdir(food_item_path):
        img_path = os.path.join(food_item_path, image_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        img = img / 255.0

        X_val.append(img)
        y_val.append(food_labels[food_item])

# Convert lists into NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

base_model = MobileNetV2(weights="imagenet",include_top=False,input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([base_model,GlobalAveragePooling2D(),Dropout(0.3),
                    Dense(128, activation="relu"),Dense(20, activation="softmax")])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=5,validation_data=(X_val, y_val),verbose=0)

print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
model.save("food_calorie_model.keras")
print("Model saved successfully!")

import pandas as pd

# Load nutrition dataset
nutrition_df = pd.read_csv("food_nutrition.csv")

# Reverse label dictionary
class_names = {
    0: "burger", 1: "butter_naan", 2: "chai",3: "chapati", 4: "chole_bhature", 5: "dal_makhani",
    6: "dhokla", 7: "fried_rice", 8: "idli", 9: "jalebi", 10: "kaathi_rolls", 11: "kadai_paneer",
    12: "kulfi", 13: "masala_dosa", 14: "momos", 15: "paani_puri", 16: "pakode", 17: "pav_bhaji", 
    18: "pizza", 19: "samosa"
}

# User image
image_path = input("Enter image path: ")
img = cv2.imread(image_path)

if img is None:
    print("Invalid image path!")

else:
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = img.reshape(1,224,224,3)
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    food_name = class_names[predicted_class]

    print("PREDICTION")
    print(f"Food Item : {food_name}")
    print(f"Confidence : {confidence*100:.2f}%")

    food_info = nutrition_df[
        nutrition_df["Food"].str.lower() == food_name.lower()
    ]

    if not food_info.empty:
        serving_size = food_info.iloc[0]["Serving Size"]
        calories = food_info.iloc[0]["Calories (kcal)"]
        protein = food_info.iloc[0]["Protein (g)"]
        carbs = food_info.iloc[0]["Carbs (g)"]
        fat = food_info.iloc[0]["Fat (g)"]

        print("NUTRITION")
        print(f"Serving Size  : {serving_size}")
        print(f"Calories : {calories} kcal")
        print(f"Protein  : {protein} g")
        print(f"Carbs    : {carbs} g")
        print(f"Fat      : {fat} g")

    else:
        print("Nutrition information not found.")