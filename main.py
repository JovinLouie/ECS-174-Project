import torch
import torch.optim as optim
from unet import UNet
from dataset import get_dataloaders
from train import train_model
from eval import evaluate

def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 8
    learning_rate = 0.0005
    epochs = 20
    image_size = 400
    n_class = 1  # Binary segmentation (road vs background)
    
    # Enable data augmentation, we have plenty of data so probably not necessary
    use_augmentation = False
    
    print('Loading data...')
    trainloader, valloader, testloader = get_dataloaders(
        root_dir='./data',
        batch_size=batch_size,
        augmentation=use_augmentation,
        image_size=image_size
    )
    
    print(f'Training samples: {len(trainloader.dataset)}')
    print(f'Validation samples: {len(valloader.dataset)}')
    print(f'Test samples: {len(testloader.dataset)}')
    
    # Initialize model
    print('\nInitializing model...')
    model = UNet(n_class=n_class)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Optimizer, haven't considered other options yet
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print(f'\nStarting training for {epochs} epochs...')
    print('-' * 60)
    
    train_losses, train_acc, val_acc = train_model(
        model=model,
        trainloader=trainloader,
        testloader=valloader,  # Using valloader for validation during training
        title='Road Segmentation UNet',
        optimizer=optimizer,
        epochs=epochs,
        device=device
    )
    
    # Final evaluation on test set
    print('\n' + '=' * 60)
    print('Evaluating on test set...')
    test_accuracy = evaluate(model, testloader, device)
    print(f'Final Test Dice Coefficient: {test_accuracy:.2f}%')
    print('=' * 60)
    
    # Save final metrics
    with open('training_results.txt', 'w') as f:
        f.write('Road Segmentation Training Results\n')
        f.write('=' * 60 + '\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Batch Size: {batch_size}\n')
        f.write(f'Learning Rate: {learning_rate}\n')
        f.write(f'Image Size: {image_size}x{image_size}\n')
        f.write(f'Data Augmentation: {use_augmentation}\n')
        f.write(f'\nFinal Train Dice: {train_acc[-1]*100:.2f}%\n')
        f.write(f'Final Val Dice: {val_acc[-1]:.2f}%\n')
        f.write(f'Test Dice: {test_accuracy:.2f}%\n')
    
    print('\nTraining complete! Results saved to training_results.txt')
    print('Model checkpoint saved to unet_checkpoint.pth')
    print('Training plots saved to Road_Segmentation_UNet.png')


if __name__ == '__main__':
    main()