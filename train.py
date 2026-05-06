import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("employee attrition dataset.csv")

# Convert target
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# Split features
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Columns
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])

# Train
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(x_train, y_train)

# Save
joblib.dump(pipeline, "pipeline.pkl")

print("✅ pipeline.pkl created successfully!")