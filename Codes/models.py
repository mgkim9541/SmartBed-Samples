import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_dim: int):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model

def train_model(X_train, y_train, epochs=50, batch_size=32):
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    return model
