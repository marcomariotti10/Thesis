from tensorflow import keras as tfkl

def build_model(input_shape):
    # Input layer for the neural network
    grid_map_input = tfkl.Input(shape=input_shape, name='grid_map_input')

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

    # Create the model
    model = tfkl.Model(inputs=grid_map_input, outputs=decoded)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model