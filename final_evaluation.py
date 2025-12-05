# final_evaluation.py
# Jovin Louie

import random
import torch
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from unet import UNet
from eval import evaluate
import os


def plot_test_dice(test_dice: float, title: str = "Final Test Set Dice", output_dir: str = "final_eval_outputs"):
    # Simple bar plot of the test Dice score.
    plt.figure(figsize=(4, 6))
    plt.bar(["Test Dice"], [test_dice])
    plt.ylim(0, 100)
    plt.ylabel("Dice (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, (title.replace(" ", "_") + ".png")))
    plt.close()

# Undo normalization: image is a tensor [C, H, W].
def denormalize(image):
    mean_dataset = [0.485, 0.456, 0.406]
    std_dataset = [0.229, 0.224, 0.225]
    
    mean = torch.tensor(mean_dataset).view(-1, 1, 1)
    std = torch.tensor(std_dataset).view(-1, 1, 1)
    image = image * std + mean
    return image.clamp(0, 1)

def visualize_sample(image, gt_mask, pred_mask, idx: int, output_dir: str):
    # Visualize a single sample: input image, ground truth mask, predicted mask.
    # image: tensor [C, H, W]
    # gt_mask: tensor [1, H, W] or [H, W]
    # pred_mask: tensor [1, H, W] or [H, W]

    # Move to CPU just in case
    image = denormalize(image.cpu())
    gt_mask = gt_mask.cpu()
    pred_mask = pred_mask.cpu()

    # Convert shapes for plotting
    # image: [H, W, C]
    img_np = image.permute(1, 2, 0).numpy()

    # masks: [H, W]
    gt_np = gt_mask.squeeze().numpy()
    pred_np = pred_mask.squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_np, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    fig.suptitle(f"Test Sample #{idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"test_sample_{idx}.png"))
    plt.close()


def run_final_evaluation(
    model_path: str = "unet_checkpoint-40_Epochs.pth",
    batch_size: int = 4,
    # num_workers: int = 2,
    num_random_samples: int = 10,
    image_size: int = 400, # 400x400 images
    output_dir: str = "final_eval_outputs",
    test_dice: bool = True,
    sample_indices: list[int] | None = None
):
    
    # Run the final evaluation on the test set, plot metrics, and show samples.

    # model_path: path to the saved UNet checkpoint (e.x. 'unet_checkpoint.pth'
    #             or 'unet_checkpoint-40_Epochs.pth').

    os.makedirs(output_dir, exist_ok=True)

    print(f"Using model: {model_path}")

    # Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data (train/val unused here but returned by get_dataloaders) 
    trainloader, valloader, testloader = get_dataloaders(
        root_dir="./data",
        batch_size=batch_size,
        augmentation=False,
        image_size=image_size,
    )
    test_dataset = testloader.dataset
    print(f"Test samples: {len(test_dataset)}")

    # Load model
    n_class = 1  # binary segmentation (road vs background), same as in main.py
    model = UNet(n_class=n_class)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    if test_dice:
        # Evaluate model on test set 
        print("Evaluating model on test set...")
        test_dice = evaluate(model, testloader, device)
        print(f"Final Test Dice: {test_dice:.2f}%")

        # Plot final evaluation Dice score 
        print("Plotting final test Dice...")
        plot_test_dice(test_dice, title="Final Test Set Dice")

    # Pick random samples to visualize 
    if len(test_dataset) == 0:
        print("No samples in test set; skipping visualization.")
        return

    # num_samples = min(num_random_samples, len(test_dataset))
    # print(f"Plotting {num_samples} random samples from the test set...")
    # indices = random.sample(range(len(test_dataset)), k=num_samples)

    # For non-random samples
    if sample_indices is not None and len(sample_indices) > 0:
        print(f"Using user-provided sample indices: {sample_indices}")

        # Filter out invalid indices
        indices = [i for i in sample_indices if 0 <= i < len(test_dataset)]
        if len(indices) == 0:
            print("No valid indices provided. Aborting visualization.")
            return
    else:
        # Fallback to random sampling
        num_samples = min(num_random_samples, len(test_dataset))
        print(f"Selecting {num_samples} random samples from the test set...")
        indices = random.sample(range(len(test_dataset)), k=num_samples)

    model.eval()
    with torch.no_grad():
        for idx in indices:
            image, gt_mask = test_dataset[idx]
            image_batch = image.unsqueeze(0).to(device)

            logits = model(image_batch)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()

            # pred_mask shape: [1, 1, H, W]; squeeze batch dim
            pred_mask = pred_mask.squeeze(0)

            visualize_sample(image, gt_mask, pred_mask, idx, output_dir)


if __name__ == "__main__":
    # run_final_evaluation(model_path="unet_checkpoint-20_Epochs.pth", test_dice=False, sample_indices=[522, 1256, 1385])
    run_final_evaluation()
