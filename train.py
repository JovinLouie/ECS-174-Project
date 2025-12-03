import torch
from eval import evaluate
from dice_coeff import dice_coeff
from plot import plot_results

# Dice Loss = 1 - Dice Coefficient
# Penalizes missing thin structures, helps w/ class imbalance
def dice_loss(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)  # convert raw model pred to probabilities (0, 1)
    pred = pred.float()
    target = target.float() # ground truth mask
    
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()  # Dice loss

# BCE treats all pixels equally, so predicting all background -> low loss if road is only 10% of image
# Combine BCE and Dice Loss for better performance on imbalanced data
# bce_weight: weight for BCE, (1 - bce_weight) for Dice, lower for thin roads
def bce_dice_loss(pred, target, bce_weight=0.5):
    # Calculate positive weight based on class imbalance
    pos_weight = (1 - target.mean()) / (target.mean() + 1e-6) # handles class imbalance by increasing penalty for road pixels
    pos_weight = torch.clamp(pos_weight, 1.0, 5.0)  # clamp to prevent extreme gradients
    
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target, pos_weight=pos_weight
    )
    d_loss = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * d_loss


def train_model(model, trainloader, testloader, title, optimizer=None, epochs=1, device='cpu'):
    model.to(device)
    loss_func = bce_dice_loss
    train_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for batch_idx, data in enumerate(trainloader): # batch_idx was for debugging
            inputs, labels = data[0].to(device), data[1].to(device).float() # get the inputs
            
            optimizer.zero_grad() # zero the parameter gradients
            outputs = model(inputs) # forward + backward + optimize
            
            loss = loss_func(outputs, labels, bce_weight=0.5)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # calc training accuracy: use probabilities -> binary -> Dice coeff
            predicted_probs = torch.sigmoid(outputs)
            predicted_bin = (predicted_probs > 0.5).float()
            running_acc += dice_coeff(predicted_bin, labels).item()
            n_batches += 1

        # calc epoch data
        epoch_train_loss = running_loss / n_batches
        epoch_train_acc = running_acc / n_batches
        epoch_test_acc = evaluate(model, testloader, device)
        
        train_losses.append(epoch_train_loss)
        train_acc.append(epoch_train_acc * 100)  # Convert to percentage
        test_acc.append(epoch_test_acc)

        # epoch_train_acc is a fraction (0..1), epoch_test_acc already is percentage
        print('Epoch %d: Train Loss: %.3f, Train Dice: %.2f%%, Val Dice: %.2f%%' %
              (epoch + 1, epoch_train_loss, epoch_train_acc * 100, epoch_test_acc))
    
    print('Finished Training')
    plot_results(train_losses, train_acc, test_acc, title)
    torch.save(model.state_dict(), "unet_checkpoint.pth")
    return train_losses, train_acc, test_acc