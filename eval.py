import torch
from dice_coeff import dice_coeff

# Evaluate Model On Test Set Fucntion
def evaluate(model, testloader, device):
    total = 0
    batches = 0
    
    model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device).float()
            outputs = model(inputs)

            # logits to probabilities to binary mask
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()

            # dice coefficient
            batch_dice = dice_coeff(predicted, labels).item()
            total += batch_dice
            batches += 1
    
    return 100*(total / batches)