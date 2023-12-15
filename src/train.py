import tensorflow as tf
import numpy as np
from dvclive import Live
import os
import json
import yaml
import pickle
def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    layers_count = params["layers_count"]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train = np.load('data/iris/X_train.npy')
    X_test = np.load('data/iris/X_test.npy')
    y_train = np.load('data/iris/y_train.npy')
    y_test = np.load('data/iris/y_test.npy')
    history = model.fit(X_train, y_train, epochs=100, batch_size=10,validation_data=(X_test, y_test),verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    metrics = {
    "accuracy": test_acc,
    "loss": test_loss
    }
    with Live() as live:
        live.log_metric("accuracy",test_acc)
        live.log_metric("loss", test_loss)
    with open("metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)
    os.makedirs('data/trained',exist_ok=True)
    with open('data/trained/history.pkl', 'wb') as f:
        pickle.dump(history, f)
if __name__ == "__main__":
    main()
