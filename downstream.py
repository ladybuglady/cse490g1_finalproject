''' This module evaluates the performance of a trained CPC encoder '''

from decode_ecog_data_utils import ECoGGenerator
from os.path import join, basename, dirname, exists
import keras
import tensorflow as tf



def build_model(encoder_path, learning_rate):

    # Read the encoder
    encoder = keras.models.load_model(encoder_path)
    print(encoder.summary())

    # Freeze weights
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # Define the classifier
    event_shape = (64, 500)
    x_input = keras.layers.Input(event_shape)
    x = encoder(x_input)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=2, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=(x_input), outputs=x)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model


def downstream_model(encoder_path, epochs, batch_size, output_dir, lr=1e-4):

    # Prepare data
    train_data = ECoGGenerator(batch_size, subset='train', rescale=False) # may need to change this since 

    validation_data = ECoGGenerator(batch_size, subset='valid', rescale=False)

    # Prepares the model
    model = build_model(encoder_path, learning_rate=lr)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]

    # Trains the model
    fittedModel = model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )
    
    print(fittedModel.history.keys())
    # summarize history for accuracy
    plt.plot(fittedModel.history['binary_accuracy'])
    plt.plot(fittedModel.history['val_binary_accuracy'])
    plt.title('Wrist Movement Decode Accuracy for subject a0f66459')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1.1])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(fittedModel.history['loss'])
    plt.plot(fittedModel.history['val_loss'])
    plt.title('Wrist Movement Decode Accuracy a0f66459')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim([0, 1.1])
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Saves the model
    #model.save(join(output_dir, 'supervised.h5'))


if __name__ == "__main__":

    downstream_model(
        encoder_path='cpc_ecog_conv1d_20min/encoder.h5',
        epochs=10,
        batch_size=32,
        output_dir='models/cpc_ecog_conv1d_20min',
        lr=1e-3,
    )
