import pandas as pd
house = pd.read_csv('enhanced_house_price_dataset.csv')
# print(house[0:10])

# Selecting input features and target variable
X = house[['Area','Bedrooms','Bathrooms']]
y = house['Price']
# print(X,y)

# Splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20)

# Model selection and training
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

# Making predictions on test data
y_pred = lr.predict(X_test)
# print(y_pred)

# Evaluating model performance
from sklearn.metrics import r2_score,mean_squared_error
print("r2_score : ",r2_score(y_test,y_pred)*100)
print("MSE score :",mean_squared_error(y_test,y_pred))

# Taking user input for new house prediction
area = int(input("Enter the Area(in sqft) :"))
bed_rooms = int(input("Enter the no. of Bedrooms :"))
bath_rooms = int(input("Enter the no. of Bathrooms :"))
new_data = pd.DataFrame({
    'Area':[area],'Bedrooms':[bed_rooms],'Bathrooms':[bath_rooms]
})
new_house = lr.predict(new_data)
print("Price prediction for new home :",new_house)

# Data visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.scatter(X_test['Area'],y_test,label="Actual Price")
plt.scatter(X_test['Area'],y_pred,label="Predicted Price")
plt.scatter(area,new_house[0],label="New House Price Prediction",marker='d',color="green",s=120)

plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.title("Area vs House Price Prediction")
plt.legend()
plt.show()