import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as tfl

    #Load dataset from folder
path = 'C:\MachLearnProg\Villain\Villains'


    #Pre-Process and augment data using horizontal flipping
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                                rescale=1./255,
                                                                validation_split=0.2)

    #80% of dataset used for training
train_ds = train_datagen.flow_from_directory(
        path,
        subset='training',
        target_size=(256 , 256),
        batch_size=2)

    #20% of dataset used for validation
val_ds = train_datagen.flow_from_directory(
        path,
        subset='validation',
        target_size=(256 , 256),
        batch_size=2 )

    #Simple CNN model
def convolutional_model(input_shape):
    input_img = tf.keras.Input(shape=input_shape)
    x = tfl.Conv2D(32, kernel_size=(4,4), strides=(1,1), padding='SAME')(input_img)
    x = tfl.ReLU()(x)
    x = tfl.MaxPool2D(pool_size=(8,8), strides=(8,8), padding='SAME')(x)
    x = tfl.Conv2D(16, kernel_size=(4,4), strides=(1,1), padding='SAME')(x)
    x = tfl.ReLU()(x)
    x = tfl.MaxPool2D(pool_size=(4,4), strides=(4,4), padding='SAME')(x)
    x = tfl.Flatten()(x)
    outputs = tfl.Dense(units=5, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


conv_model = convolutional_model((256, 256, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
history = conv_model.fit(train_ds, epochs=20, validation_data=val_ds)

conv_model.save("VillainClassifier.h5")