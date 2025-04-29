import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

## 1. Data Preprocessing and Model Building
# Set random seed for reproducibility
np.random.seed(42)

## 2. Load the Dataset
# Load the data
loan_data = pd.read_csv('loan_data - copy.csv')
data_dict = pd.read_csv('Data_Dictionary.csv')

# Display basic info
print("Dataset shape:", loan_data.shape)
print("\nFirst few rows:")
loan_data.head()

## 3. Check for Null Values
# Check for null values
null_values = loan_data.isnull().sum()
null_percentage = (loan_data.isnull().sum() / len(loan_data)) * 100

print("Null values in dataset:")
pd.DataFrame({'Null Count': null_values, 'Percentage': null_percentage}).sort_values(by='Percentage', ascending=False).head(20)

## 4. Percentage of Defaults
# Calculate default rate
default_rate = loan_data['TARGET'].mean()
print(f"Percentage of defaults (TARGET=1): {default_rate*100:.2f}%")

# Plot the target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='TARGET', data=loan_data)
plt.title('Loan Repayment Status (0=Repaid, 1=Default)')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

## 5. Balance the Dataset
# Separate features and target
X = loan_data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = loan_data['TARGET']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

print(f"Original dataset shape: {X_processed.shape}")
print(f"Resampled dataset shape: {X_resampled.shape}")

## 6. Plot the Balanced Data
# Plot the balanced data
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title('Balanced Loan Repayment Status (0=Repaid, 1=Default)')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

## 7. Train-Test Split
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Convert to categorical for neural network
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

## 8. Build Deep Learning Model
# Get input dimension
input_dim = X_train.shape[1]

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1)

## 9. Model Evaluation
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Predict probabilities
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate Sensitivity (Recall)
sensitivity = recall_score(y_test, y_pred)
print(f"Sensitivity (Recall): {sensitivity:.4f}")

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
print(f"ROC AUC Score: {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
