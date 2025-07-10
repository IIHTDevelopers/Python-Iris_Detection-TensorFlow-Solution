import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load and preprocess dataset
def load_and_preprocess():
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # 1 if Setosa, else 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Step 2: Build model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    return model

# Step 4: Evaluate model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

# Step 5: Predict a sample
def predict_sample(model, scaler):
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Known Setosa
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    print(f"Predicted probability of being Setosa: {prediction[0][0]:.4f}")

# Step 6: Main function
def main():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    model = build_model()
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    predict_sample(model, scaler)

# Run
if __name__ == "__main__":
    main()
