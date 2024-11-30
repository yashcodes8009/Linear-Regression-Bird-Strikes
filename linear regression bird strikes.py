import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
import random

# Load dataset
data = pd.read_csv('C:/Users/micro/Downloads/Bird_strikes.csv')

# Drop unnecessary columns
data = data.drop(columns=['RecordID', 'FlightDate', 'Remarks'], errors='ignore')

# Drop rows with missing values in the target column ('Damage')
data = data.dropna(subset=['Damage'])

# Drop rows with any other missing values
data = data.dropna()

# Convert categorical variables to dummy variables
dum = pd.get_dummies(data, columns=['AircraftType', 'WildlifeSpecies', 'ConditionsPrecipitation'], dtype='int64')

# Function to handle non-numeric values like '11 to 100'
def convert_range(value):
    if isinstance(value, str) and 'to' in value:
        low, high = value.split(' to ')  # Split by 'to'
        return (float(low) + float(high)) / 2  # Return the midpoint of the range
    else:
        return float(value)  # Direct conversion for numeric values

# Apply the conversion to 'Altitude' and 'NumberStruck' columns
dum['Altitude'] = dum['Altitude'].apply(convert_range)
dum['NumberStruck'] = dum['NumberStruck'].apply(convert_range)

# Check and fix the target column 'Damage' if non-numeric
if dum['Damage'].dtype == 'object':
    dum['Damage'] = dum['Damage'].map({'Minor': 0, 'Major': 1})  # Adjust mapping as needed

# Define independent variables
x = dum[['Altitude', 'NumberStruck', 'AircraftType_Airplane', 
         'WildlifeSpecies_American crow', 'ConditionsPrecipitation_Rain']]

# Define dependent variable
y = dum['Damage']

# Split data into train and test sets
random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 1. Linear Regression
print("\n--- Linear Regression ---")
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

slope = reg.coef_
intercept = reg.intercept_
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Slope (Coefficients):", slope)
print("Intercept:", intercept)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
print("Mean Absolute Error (MAE):", mae)

# 2. KNN Model
print("\n--- K-Nearest Neighbors (KNN) ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", knn_accuracy)

# 3. Logistic Regression
print("\n--- Logistic Regression ---")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train, y_train)
y_pred_log = log_reg.predict(x_test)

log_reg_accuracy = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", log_reg_accuracy)

# Visualization Example (Scatterplot for Linear Regression)
sns.scatterplot(x=x_test['Altitude'], y=y_test, label='Actual')
sns.scatterplot(x=x_test['Altitude'], y=y_pred, label='Predicted', marker='x')
plt.title('Scatterplot of Altitude vs Damage (Linear Regression)')
plt.legend()
plt.show()


knn_accuracy = accuracy_score(y_test, y_pred_knn) * 100
log_reg_accuracy = accuracy_score(y_test, y_pred_log) * 100

