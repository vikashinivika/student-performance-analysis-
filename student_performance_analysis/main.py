import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D data = {
'Attendance': [80, 85, 90, 70, 60, 75, 95, 85, 65, 70],
'Study_Hours': [12, 15, 10, 8, 6, 9, 14, 13, 7, 6],
'Performance': [85, 90, 88, 70, 60, 75, 95, 85, 65, 70]
}
df = pd.DataFrame(data) plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(df['Attendance'], df['Performance'], color='blue', alpha=0.7) plt.title('Attendance vs Performance')
plt.xlabel('Attendance (%)') plt.ylabel('Performance (marks)')

# Study Hours vs Performance plt.subplot(1, 2, 2)
plt.scatter(df['Study_Hours'], df['Performance'], color='green', alpha=0.7) plt.title('Study Hours vs Performance')
plt.xlabel('Study Hours (hrs)') plt.ylabel('Performance (marks)')

plt.tight_layout()
 
plt.show()
# Train-Test Split & Scaling
X = df[['Attendance', 'Study_Hours']] # Features y = df['Performance']	# Target

scaler = StandardScaler() X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.2, random_state=42
)
# Train Model
model = LinearRegression() model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test) # Evaluation
mse = mean_squared_error(y_test, y_pred) rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("===== Model Evaluation =====") print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}") print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}") print("\n===== Regression Coefficients =====") print(f"Intercept: {model.intercept_:.2f}") print(f"Coefficient for Attendance: {model.coef_[0]:.2f}") print(f"Coefficient for Study Hours: {model.coef_[1]:.2f}")
 

# 3D Visualization of Regression Plane fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of actual data
ax.scatter(df['Attendance'], df['Study_Hours'], df['Performance'], color='blue', label='Data Points')

# Prediction surface
attendance_range = np.linspace(df['Attendance'].min(), df['Attendance'].max(), 20) study_hours_range = np.linspace(df['Study_Hours'].min(), df['Study_Hours'].max(), 20) attendance_grid, study_hours_grid = np.meshgrid(attendance_range, study_hours_range)

grid_scaled = scaler.transform( np.c_[attendance_grid.ravel(), study_hours_grid.ravel()]
)
performance_grid = model.predict(grid_scaled).reshape(attendance_grid.shape)


surf = ax.plot_surface(attendance_grid, study_hours_grid, performance_grid, color='red', alpha=0.5)

ax.set_xlabel('Attendance (%)') ax.set_ylabel('Study Hours (hrs)') ax.set_zlabel('Performance (marks)')
ax.set_title('Regression Model: Attendance & Study Hours vs Performance')

fig.colorbar(surf) plt.show()
