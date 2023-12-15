import matplotlib.pyplot as plt
import pickle
import os
from dvclive import Live


def main():
    with open('data/trained/history.pkl', 'rb') as f:
        history = pickle.load(f)

    datapoints = []
    for epoch in range(100):
        datapoint = {
            'epoch': epoch + 1,
            'loss': history.history['loss'][epoch],
            'accuracy': history.history['accuracy'][epoch],
            'val_loss': history.history['val_loss'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch]
        }
        datapoints.append(datapoint)

    with Live() as live:
        live.log_plot("iris",datapoints,x="epoch",y="loss",title="Training and validation loss")
    # loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='b')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='r')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs('eval/plots/images',exist_ok=True)
    plt.savefig('eval/plots/images/loss.png')
if __name__ == "__main__":
    main()
