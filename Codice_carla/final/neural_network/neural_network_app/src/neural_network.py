from constants import *  # type: ignore
import tensorflow as tf
import tensorflow.keras.layers as tfkl

def build_model(input_shape):
    grid_map_input = tfkl.Input(input_shape, name='grid_map_input')

    # Encoder
    x = tfkl.Conv2D(32, (3, 3), activation='relu', padding='same')(grid_map_input)
    x = tfkl.MaxPooling2D((2, 2), padding='same')(x)
    x = tfkl.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tfkl.MaxPooling2D((2, 2), padding='same')(x)
    x = tfkl.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = tfkl.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = tfkl.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = tfkl.UpSampling2D((2, 2))(x)
    x = tfkl.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tfkl.UpSampling2D((2, 2))(x)
    x = tfkl.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tfkl.UpSampling2D((2, 2))(x)
    decoded = tfkl.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs=grid_map_input, outputs=decoded)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

if __name__ == "__main__":
    shape_input = (X_RANGE, Y_RANGE, 1)  # type: ignore
    model = build_model(shape_input)
    model.summary()