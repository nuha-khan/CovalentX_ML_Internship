import os
import cv2

cat_path = "test_set/cats"
dog_path = "test_set/dogs"

X = []
y = []

# Process cat images
for image_name in os.listdir(cat_path):           # Gets all files inside folder
    img_path = os.path.join(cat_path,image_name)  # Create full image path
    img = cv2.imread(img_path)                    # Reads image
    img = cv2.resize(img,(48,48))                 # Resize image as SVM requires same-size input
    img = img.flatten()                           # Flatten image as SVM expects feature vectors not image matrices
    X.append(img)                                 # Store image in dataset
    y.append(0)                                   # Add label

# Process dog images
for image_name in os.listdir(dog_path):           
    img_path = os.path.join(dog_path,image_name)
    img = cv2.imread(img_path)                    
    img = cv2.resize(img,(48,48))                    
    img = img.flatten()                           
    X.append(img)                                 
    y.append(1)                                   

# Convert lists into NumPy arrays
import numpy as np
X = np.array(X)
y = np.array(y)
print("Data :",X)
print("Class Labels :",y)
X = X / 255.0

# Splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model selection and training
from sklearn.svm import SVC
sv_model = SVC(kernel="rbf")
sv_model.fit(X_train, y_train)

# Making predictions on test data
predictions = sv_model.predict(X_test)

# Evaluating model performance
from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(y_test,predictions)*100)

# Real prediction on unseen image
user_image = input("Enter image path: ")
img = cv2.imread(user_image)
img = cv2.resize(img,(48,48))
img = img.flatten()
img = img / 255.0
img = img.reshape(1,-1)
prediction = sv_model.predict(img)

if prediction[0] == 0:
    print("The image is of a Cat")
else:
    print("The image is of a Dog")