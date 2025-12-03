# Computes similarity between two sets (binary masks), common for segmentation tasks
# Values range from 0 (no overlap) to 1 (perfect overlap)
# Does not care about true negatives, focuses on road structure, good since roads may take up low percentage of image
def dice_coeff(pred, target, epsilon=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean()