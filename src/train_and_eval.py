from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import hamming_loss, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def train_and_evaluate(model, X_train, Y_train, X_test, Y_test):
    model.compile(
        optimizer=Adam(0.0005),
        loss={'posture': 'categorical_crossentropy', 'comfort': 'categorical_crossentropy'},
        metrics={'posture': ['accuracy'], 'comfort': ['accuracy']}
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, Y_train,
        epochs=15, batch_size=2048,
        validation_split=0.15,
        callbacks=[early_stop], verbose=1
    )

    eval_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Posture Accuracy: {eval_results[3]:.4f}")
    print(f"Comfort Accuracy: {eval_results[4]:.4f}")
    return history, eval_results

def analyze_performance(model, X_test, Y_test, history):
    y_pred = model.predict(X_test)
    posture_pred = np.argmax(y_pred[0], axis=1)
    comfort_pred = np.argmax(y_pred[1], axis=1)
    posture_true = np.argmax(Y_test['posture'], axis=1)
    comfort_true = np.argmax(Y_test['comfort'], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(confusion_matrix(posture_true, posture_pred)).plot(ax=axes[0], values_format='d')
    axes[0].set_title("Posture Confusion Matrix")
    ConfusionMatrixDisplay(confusion_matrix(comfort_true, comfort_pred)).plot(ax=axes[1], values_format='d')
    axes[1].set_title("Comfort Confusion Matrix")
    plt.tight_layout()
    plt.show()

    h_posture = hamming_loss(posture_true, posture_pred)
    h_comfort = hamming_loss(comfort_true, comfort_pred)
    avg_hamming = (h_posture + h_comfort) / 2
    exact_match = np.mean((posture_true == posture_pred) & (comfort_true == comfort_pred))

    print(f"Hamming Loss (Posture): {h_posture:.4f}")
    print(f"Hamming Loss (Comfort): {h_comfort:.4f}")
    print(f"Average Hamming Loss: {avg_hamming:.4f}")
    print(f"Exact Match Accuracy: {exact_match:.4f}")