import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Example usage
if __name__ == '__main__':
    # Assume you have preprocessed data
    # For demonstration, we'll use dummy data:
    import numpy as np

    input_dim = 20  # replace with your actual number of features
    X_train = np.random.random((1000, input_dim))
    y_train = np.random.randint(0, 2, size=(1000, 1))

    # Build the model
    model = build_model(input_dim)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Save the model to an HDF5 file
    model.save('ids_model.h5')
    print("Model saved as ids_model.h5")
