import tensorflow as tf
import numpy as np
import yaml
def main():
    params = yaml.safe_load(open("params.yaml"))["featurize"]
    layers_count = params["layers_count"]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(layers_count, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train = np.load('data/iris/X_train.npy')
    X_test = np.load('data/iris/X_test.npy')
    y_train = np.load('data/iris/y_train.npy')
    y_test = np.load('data/iris/y_test.npy')

if __name__ == "__main__":
    main()
