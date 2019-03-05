from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Reshape
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
import Data


def build_simple(encoder_type, loss_type):
    input_data = Input(shape=(Data.X_1_.shape[1], ))
    # "encoded" is the encoded representation of the input

    x = Dense(48, activation="relu")(input_data)
    encoded = Dense(5, activation=encoder_type)(x)
    x = Dense(48, activation="relu")(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(Data.X_1_.shape[1], activation='tanh')(x)
    # this model maps an input to its reconstruction
    encoder = Model(input_data, encoded)
    autoencoder = Model(input_data, decoded)
    encoder.compile(optimizer='adadelta', loss=loss_type)
    autoencoder.compile(optimizer='adadelta', loss=loss_type)
    autoencoder.summary()
    return (autoencoder, encoder)


def build_conv(encoder_type, loss_type):
    input_data = Input(shape=Data.X_1[0].shape)

    x = Conv1D(128, 3, activation='relu', padding='same')(input_data)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Reshape((1, 12 * 64))(x)

    encoded = Dense(5, activation=encoder_type)(x)

    x = Dense(12 * 64, activation="relu")(encoded)
    x = Reshape((12, 64))(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    decoded = Conv1D(Data.X_1[0].shape[1], 3,
                     activation='tanh', padding='same')(x)

    encoder = Model(input_data, encoded)
    autoencoder = Model(input_data, decoded)

    encoder.compile(optimizer='adadelta', loss=loss_type)
    autoencoder.compile(optimizer='adadelta', loss=loss_type)
    autoencoder.summary()
    return (autoencoder, encoder)


def fit_all_simple(encoder_type, loss_type):
    (autoencoder, encoder) = build_simple(encoder_type, loss_type)
    autoencoder.fit(Data.X_1_, Data.X_1_,
                    epochs=1000,
                    batch_size=512,
                    shuffle=True,
                    verbose=1)
    autoencoder.save("autoencoder_simple_"+encoder_type+".h5")
    encoder.save("encoder_simple_"+encoder_type+".h5")


def fit_all_conv(encoder_type, loss_type):
    (autoencoder, encoder) = build_conv(encoder_type, loss_type)
    autoencoder.fit(Data.X_1, Data.X_1,
                    epochs=1000,
                    batch_size=512,
                    shuffle=True,
                    verbose=1)
    autoencoder.save("autoencoder_conv_"+encoder_type+".h5")
    encoder.save("encoder_conv_"+encoder_type+".h5")


fit_all_simple("relu", "mean_squared_error")
fit_all_simple("softmax", "mean_squared_error")

fit_all_conv("relu", "mean_squared_error")
fit_all_conv("softmax", "mean_squared_error")
