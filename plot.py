import matplotlib.pyplot as plt

def plot_results(train_losses, train_acc, test_acc, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(train_losses) + 1)
    # Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    # Training Accuracy
    axes[1].plot(epochs, train_acc, 'g-', label='Train Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{title} - Train Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    # Test Accuracy
    axes[2].plot(epochs, test_acc, 'r-', label='Test Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title(f'{title} - Test Accuracy')
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')