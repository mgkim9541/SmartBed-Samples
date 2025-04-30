from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Bidirectional, concatenate

def build_cnn_lstm_model():
    cnn_input = Input(shape=(16, 10, 1), name='CNN_Input')
    lstm_input = Input(shape=(10, 16), name='LSTM_Input')

    x = Conv2D(3, (2, 2), activation='relu')(cnn_input)
    x = MaxPooling2D((2, 2))(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    cnn_output = Dense(32)(x)

    y = Bidirectional(LSTM(256, return_sequences=True))(lstm_input)
    y = Dropout(0.2)(y)
    y = Bidirectional(LSTM(128, return_sequences=True))(y)
    y = Flatten()(y)
    lstm_output = Dense(32, activation='relu')(y)

    merged = concatenate([cnn_output, lstm_output])
    z = Dense(256, activation='relu')(merged)
    z = Dropout(0.2)(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(16, activation='relu')(z)

    posture_output = Dense(9, activation='softmax', name='posture')(cnn_output)
    comfort_output = Dense(4, activation='softmax', name='comfort')(z)

    return Model(inputs=[cnn_input, lstm_input], outputs=[posture_output, comfort_output])