import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

def preprocess_dataset(path="data/Sample Data.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["Case", "Pos", "Com"]).to_numpy()
    Y = df[["Pos", "Com"]].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train = {
        "CNN_Input": X_train_raw.reshape((-1, 16, 10, 1)),
        "LSTM_Input": X_train_raw.reshape((-1, 10, 16))
    }
    X_test = {
        "CNN_Input": X_test_raw.reshape((-1, 16, 10, 1)),
        "LSTM_Input": X_test_raw.reshape((-1, 10, 16))
    }
    Y_train = {
        "posture": to_categorical(Y_train_raw["Pos"]),
        "comfort": to_categorical(Y_train_raw["Com"])
    }
    Y_test = {
        "posture": to_categorical(Y_test_raw["Pos"]),
        "comfort": to_categorical(Y_test_raw["Com"])
    }

    return X_train, X_test, Y_train, Y_test