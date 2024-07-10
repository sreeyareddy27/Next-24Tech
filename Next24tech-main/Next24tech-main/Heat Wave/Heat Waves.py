import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

np.random.seed(42)
dates = pd.date_range(start='1/1/2020', periods=365, freq='D')
temperatures = np.random.normal(loc=30, scale=5, size=len(dates))
humidity = np.random.normal(loc=50, scale=10, size=len(dates))
wind_speed = np.random.normal(loc=10, scale=3, size=len(dates))

weather_data = pd.DataFrame({
    'Date': dates,
    'Temperature': temperatures,
    'Humidity': humidity,
    'WindSpeed': wind_speed
})

heatwave_threshold = 35
weather_data['IsHeatwave'] = weather_data['Temperature'].rolling(window=3).mean() > heatwave_threshold
weather_data = weather_data.dropna()

X = weather_data[['Temperature', 'Humidity', 'WindSpeed']]
y = weather_data['IsHeatwave']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
