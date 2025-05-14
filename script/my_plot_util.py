# my_plot.py
import matplotlib.pyplot as plt
import os

class Plot_graph:
    def __init__(self, save_path="./save_path"):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        self.save_path = save_path

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

        epochs = range(1, epoch + 2)

        # Loss plot
        self.axes[0].clear()
        self.axes[0].plot(epochs, self.train_losses, label="Train Loss", marker="o")
        self.axes[0].plot(epochs, self.val_losses, label="Val Loss", marker="o")
        self.axes[0].set_title("Loss")
        self.axes[0].set_xlabel("Epoch")
        self.axes[0].set_ylabel("Loss")
        self.axes[0].legend()
        self.axes[0].grid(True)

        # Accuracy plot
        self.axes[1].clear()
        self.axes[1].plot(epochs, self.train_accuracies, label="Train Accuracy", marker="o")
        self.axes[1].plot(epochs, self.val_accuracies, label="Val Accuracy", marker="o")
        self.axes[1].set_title("Accuracy")
        self.axes[1].set_xlabel("Epoch")
        self.axes[1].set_ylabel("Accuracy")
        self.axes[1].legend()
        self.axes[1].grid(True)

        plt.tight_layout()
        plt.pause(0.1)

    def save_and_close(self, interrupt=False):
        path = self.save_path.replace(".png", "_interrupt.png") if interrupt else self.save_path
        plt.ioff()
        plt.savefig(path)
        plt.close(self.fig)