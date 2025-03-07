"""
BrainSeg2D: A library for 2D brain slice segmentation using PyTorch and
segmentation_models_pytorch. It handles multi-channel TIFF images, applies
augmentations, trains a UNet-based model, and generates segmentation masks
and ROI files for ImageJ/Fiji.

Requires:
    - Python 3.x
    - PyTorch
    - segmentation_models_pytorch
    - albumentations
    - tifffile
    - scikit-image
    - opencv-python
    - roifile
    - tqdm
    - matplotlib

Example:
    from brainseg2d import train_segmentation_workflow, predict_unseen_images

    # Train a model
    model = train_segmentation_workflow(
        images_dir="path/to/train_images",
        masks_dir="path/to/train_masks",
        train_ratio=0.8,
        # ...
    )

    # Inference on new images
    predict_unseen_images(
        unseen_images_dir="path/to/new_images",
        masks_dir="path/to/train_masks",  # so we can infer num_classes
        output_masks_dir="path/to/output_masks",
        output_rois_dir="path/to/output_rois",
        model_path="path/to/model_checkpoint.pth"
    )
"""

import os
import glob
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import albumentations as A
import tifffile
from tqdm.auto import tqdm
from skimage.measure import label, regionprops
import zipfile
import roifile


# Detect whether a CUDA device is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------
# 1) Metadata & Utility Functions
# ------------------------------------------------------------------------------

def get_image_metadata(images_dir: str) -> tuple:
    """
    Extract metadata from multi-channel TIFF images in a directory.

    Inspects all .tif files in 'images_dir' and finds:
      - The first valid image's height and width (H, W).
      - The minimum number of channels among all .tif files (min_channels).

    Args:
        images_dir (str): Path to a directory containing .tif images,
                          each assumed to have shape (C, H, W).

    Returns:
        (int, int, int):
            orig_height: The height (H) of the images.
            orig_width:  The width (W) of the images.
            min_channels: The smallest channel count found among images.
        
    Raises:
        FileNotFoundError: If no TIFF images are found.
    """
    image_paths = glob.glob(os.path.join(images_dir, "*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No .tif images found in {images_dir}.")

    min_channels = float('inf')
    height, width = None, None

    for img_path in image_paths:
        img = tifffile.imread(img_path)  # shape (C, H, W)
        c, h, w = img.shape

        # Initialize height/width once
        if height is None or width is None:
            height, width = h, w

        # Track the min channel count
        if c < min_channels:
            min_channels = c

    return height, width, min_channels


def get_num_classes(masks_dir: str) -> int:
    """
    Determine the number of segmentation classes from mask files.

    Scans the directory for files named 'MASK_*', reads each mask (assumed
    to be integer-labeled), and finds the largest label. The total number of
    classes is max_label + 1 (assuming 0-based labels).

    Args:
        masks_dir (str): Directory containing mask files named 'MASK_<filename>'.

    Returns:
        int: Number of classes = (max_label + 1).

    Raises:
        FileNotFoundError: If no mask files named 'MASK_*' are found,
                           or if an individual mask file fails to load.
    """
    mask_paths = glob.glob(os.path.join(masks_dir, "MASK_*"))
    if not mask_paths:
        raise FileNotFoundError(f"No mask files (MASK_*) found in {masks_dir}.")

    label_set = set()
    for mp in mask_paths:
        mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask file: {mp}")

        unique_vals = np.unique(mask)
        label_set.update(unique_vals.tolist())

    max_label = max(label_set)
    return int(max_label + 1)


def auto_rescale(img: np.ndarray, clip_percent: float = 0.5) -> np.ndarray:
    """
    Auto-rescale an image to [0..255] using percentile-based contrast stretching.

    For instance, clip_percent=0.5 means we clip the lowest 0.5% and highest 0.5% 
    pixel intensities, then stretch the rest to 0..255.

    Args:
        img (np.ndarray): Input 2D array (H, W) of any numeric type.
        clip_percent (float): Percentage of lower/higher pixels to clip.

    Returns:
        np.ndarray: The rescaled image (dtype=uint8).
    """
    img_float = img.astype(np.float32)
    low_val = np.percentile(img_float, clip_percent)
    high_val = np.percentile(img_float, 100.0 - clip_percent)

    if high_val <= low_val:
        # Degenerate case (e.g., all pixel intensities the same)
        return np.zeros_like(img, dtype=np.uint8)

    stretched = (img_float - low_val) * (255.0 / (high_val - low_val))
    img_8u = np.clip(stretched, 0, 255).astype(np.uint8)
    return img_8u


def compute_augment_multiplier(num_images: int,
                               target_min: int = 200,
                               target_max: int = 400) -> int:
    """
    Determine how many times to replicate the dataset to reach a target dataset size.

    E.g., if we have 50 images but want at least 200 total samples, 
    we multiply by 4 (since 4 * 50 = 200).

    Args:
        num_images (int): Number of original images.
        target_min (int): Minimum desired total dataset size (after augmentation).
        target_max (int): Maximum threshold. If we already exceed this, no extra replication.

    Returns:
        int: Augmentation multiplier (>=1).
    """
    if num_images >= target_max:
        return 1
    return max(1, target_min // num_images)


def determine_batch_size() -> int:
    """
    Heuristic for setting batch size based on whether CUDA is available.

    Returns:
        int: A batch size (8 if GPU, else 4).
    """
    return 8 if device == "cuda" else 4


def get_downsampled_dims(orig_height: int,
                         orig_width: int,
                         ratio: float = 0.25) -> tuple:
    """
    Compute new downsampled dimensions ensuring each is divisible by 32 
    (which helps with UNet-like networks that downsample by factors of 2).

    Args:
        orig_height (int): Original image height.
        orig_width (int): Original image width.
        ratio (float): Fraction to downsample by (e.g. 0.25 = 25% size).

    Returns:
        (int, int): The new (height, width) after rounding to nearest multiple of 32.
    """
    new_h = int(round((orig_height * ratio) / 32) * 32)
    new_w = int(round((orig_width * ratio) / 32) * 32)
    return new_h, new_w


def split_dataset(dataset: Dataset,
                  train_ratio: float = 0.8) -> tuple:
    """
    Split a PyTorch Dataset into training and validation subsets.

    Args:
        dataset (Dataset): The full dataset.
        train_ratio (float): Proportion of data to use for training.

    Returns:
        (Subset, Subset): Two PyTorch Subset objects (train_dataset, val_dataset).
    """
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


# ------------------------------------------------------------------------------
# 2) Augmentation / Conversion
# ------------------------------------------------------------------------------

def augment_and_convert(image: np.ndarray,
                        mask: np.ndarray,
                        aug_pipeline: A.Compose,
                        num_channels: int = 4) -> tuple:
    """
    Apply an Albumentations pipeline to (image, mask) and convert the result to PyTorch tensors.

    Args:
        image (np.ndarray): Raw image of shape (C, H, W).
        mask (np.ndarray):  Raw mask of shape (H, W).
        aug_pipeline (A.Compose): An Albumentations augmentation pipeline.
        num_channels (int): Number of channels to keep.

    Returns:
        (torch.Tensor, torch.Tensor):
            img_tensor of shape (C, H, W), mask_tensor of shape (H, W).
            Both are PyTorch tensors (float32, long).
    """
    # Albumentations needs (H, W, C)
    image_hw_c = np.transpose(image, (1, 2, 0))

    augmented = aug_pipeline(image=image_hw_c, mask=mask)
    aug_img = augmented["image"]   # shape (H, W, C)
    aug_mask = augmented["mask"]   # shape (H, W)

    # Keep exactly 'num_channels'
    if aug_img.shape[-1] < num_channels:
        raise ValueError(f"Augmented image has {aug_img.shape[-1]} channels; expected {num_channels}.")
    aug_img = aug_img[:, :, :num_channels]

    # Auto-rescale each channel
    rescaled = [auto_rescale(aug_img[:, :, i]) for i in range(num_channels)]
    aug_img = np.stack(rescaled, axis=-1)

    # Convert from [0..255] to [0..1]
    aug_img_float = aug_img.astype(np.float32) / 255.0

    # Back to (C, H, W)
    aug_img_float = np.transpose(aug_img_float, (2, 0, 1))

    img_tensor = torch.tensor(aug_img_float, dtype=torch.float32)
    mask_tensor = torch.tensor(aug_mask, dtype=torch.long)

    return img_tensor, mask_tensor


def get_augmentation_pipelines(down_height: int,
                               down_width: int) -> tuple:
    """
    Create Albumentations Compose pipelines for training and validation.

    Training includes random flips and slight rotation.
    Validation only resizes.

    Args:
        down_height (int): Target height for resizing.
        down_width (int):  Target width for resizing.

    Returns:
        (A.Compose, A.Compose): (train_augmentation, val_augmentation)
    """
    train_aug = A.Compose([
        A.Resize(height=down_height, width=down_width),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, p=0.5,
                 border_mode=cv2.BORDER_CONSTANT,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
    ], additional_targets={"mask": "mask"})

    val_aug = A.Compose([
        A.Resize(height=down_height, width=down_width),
    ], additional_targets={"mask": "mask"})

    return train_aug, val_aug


# ------------------------------------------------------------------------------
# 3) Dataset Class
# ------------------------------------------------------------------------------

class ImageDataset(Dataset):
    """
    PyTorch Dataset for multi-channel TIF images + segmentation masks.
    Repeats images up to 'augment_multiplier' times to expand the dataset size.
    """
    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 aug_pipeline: A.Compose = None,
                 augment_multiplier: int = 1,
                 num_channels: int = 4):
        """
        Args:
            images_dir (str): Directory of TIF images, each (C,H,W).
            masks_dir  (str): Directory of masks, named 'MASK_<same_filename>'.
            aug_pipeline (A.Compose): Albumentations pipeline for data augmentation.
            augment_multiplier (int): Factor to artificially expand the dataset.
            num_channels (int): Number of channels to keep (from the TIF).
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))

        self.aug_pipeline = aug_pipeline
        self.augment_multiplier = augment_multiplier
        self.num_channels = num_channels

        # Artificially expand by repeating the file list
        self.expanded_image_paths = self.image_paths * augment_multiplier

    def __len__(self) -> int:
        return len(self.expanded_image_paths)

    def __getitem__(self, idx: int) -> dict:
        # Map the given idx to the "real" image index
        true_idx = idx % len(self.image_paths)

        image_path = self.image_paths[true_idx]
        filename = os.path.basename(image_path)
        mask_path = os.path.join(self.masks_dir, f"MASK_{filename}")

        # Read raw TIF (C, H, W) and mask (H, W)
        raw_img = tifffile.imread(image_path)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.aug_pipeline is not None:
            img_tensor, mask_tensor = augment_and_convert(
                raw_img, raw_mask, self.aug_pipeline, self.num_channels
            )
        else:
            # Fallback: just auto-rescale each channel, no augmentation
            raw_img = raw_img[:self.num_channels, :, :]
            img_rescaled = np.stack(
                [auto_rescale(raw_img[i]) for i in range(self.num_channels)], axis=0
            )
            img_float = img_rescaled.astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_float, dtype=torch.float32)
            mask_tensor = torch.tensor(raw_mask, dtype=torch.long)

        return {
            "image": img_tensor,   # (C, H, W)
            "mask": mask_tensor,   # (H, W)
            "filename": filename
        }


# ------------------------------------------------------------------------------
# 4) Model Building & Training
# ------------------------------------------------------------------------------

def build_model(device: str,
                num_classes: int,
                num_channels: int,
                load_model: bool = False,
                model_path: str = None) -> nn.Module:
    """
    Build or load a UNet with a ResNet34 encoder from segmentation_models_pytorch.

    Args:
        device (str): 'cpu' or 'cuda'.
        num_classes (int): Number of segmentation classes (channels in output).
        num_channels (int): Number of channels in input images.
        load_model (bool): If True, load weights from 'model_path'.
        model_path (str, optional): Path to a .pth checkpoint to load.

    Returns:
        nn.Module: The UNet model on the given device.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if not load_model else None,
        in_channels=num_channels,
        classes=num_classes
    ).to(device)

    if load_model and model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_classes: int,
                device: str,
                num_epochs: int = 50,
                learning_rate: float = 1e-4,
                save_checkpoint_path: str = None) -> nn.Module:
    """
    Train and validate a segmentation model using weighted cross-entropy.

    Overwrites the same saved model file after each epoch if 'save_checkpoint_path'
    is provided. If training is interrupted (Ctrl+C), it saves and returns
    the partially-trained model.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        num_classes (int): Number of classes (including background).
        device (str): 'cpu' or 'cuda'.
        num_epochs (int): Number of training epochs (default=50).
        learning_rate (float): Learning rate (default=1e-4).
        save_checkpoint_path (str): If not None, path to save model checkpoints.

    Returns:
        nn.Module: The trained (or partially-trained) model.
    """
    # Weighted cross-entropy: background label is de-weighted to 0.1
    weights = torch.ones(num_classes, dtype=torch.float, device=device)
    weights[0] = 0.1
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Ensure checkpoint directory exists, if needed
    if save_checkpoint_path is not None:
        os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)

    try:
        for epoch in range(num_epochs):
            # ---------- TRAIN ----------
            model.train()
            total_train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for batch in train_pbar:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_pbar.set_postfix({"Loss": f"{loss.item():.2f}"})

            avg_train_loss = total_train_loss / len(train_loader)

            # ---------- VALID ----------
            model.eval()
            total_val_loss = 0.0
            correct_pixels = 0
            total_pixels = 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    logits = model(images)

                    val_loss = criterion(logits, masks)
                    total_val_loss += val_loss.item()

                    preds = torch.argmax(logits, dim=1)
                    correct_pixels += (preds == masks).sum().item()
                    total_pixels += masks.numel()

            avg_val_loss = total_val_loss / len(val_loader)
            pixel_accuracy = correct_pixels / total_pixels
            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Pixel Acc: {pixel_accuracy:.4%} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.1e}")

            # ---------- SAVE CHECKPOINT ----------
            if save_checkpoint_path is not None:
                torch.save(model.state_dict(), save_checkpoint_path)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nTraining interrupted. Saving checkpoint and returning model...")
        if save_checkpoint_path is not None:
            torch.save(model.state_dict(), save_checkpoint_path)
            print(f"Checkpoint saved to: {save_checkpoint_path}")

    return model


# ------------------------------------------------------------------------------
# 5) Visualization Functions
# ------------------------------------------------------------------------------

def plot_nchannel_image(raw_image: np.ndarray,
                        cmap_list: list = None,
                        titles: list = None,
                        clip_percent: float = 0.5) -> None:
    """
    Plot an N-channel image side-by-side, each channel with its own colormap.

    Args:
        raw_image (np.ndarray): Shape (C, H, W).
        cmap_list (list of str, optional): List of matplotlib colormap names for each channel.
        titles (list of str, optional): List of titles for each subplot.
        clip_percent (float): Percentile-based contrast stretching per channel.
    """
    num_channels = raw_image.shape[0]

    # Default colormaps if none provided
    if cmap_list is None:
        default_colormaps = ["Reds", "Greens", "Blues", "Purples", "Oranges", "Greys"]
        cmap_list = default_colormaps[:num_channels]

    # Default titles if none provided
    if titles is None:
        titles = [f"Channel {i+1}" for i in range(num_channels)]

    # Rescale each channel
    rescaled_channels = [auto_rescale(raw_image[i], clip_percent) for i in range(num_channels)]

    fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))
    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        axes[i].imshow(rescaled_channels[i], cmap=cmap_list[i])
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.show()


def visualize_dataloader_samples(dataloader: DataLoader,
                                masks_dir: str,
                                num_show: int = 2,
                                clip_percent: float = 0.5) -> None:
    """
    Visualize a few samples from a DataLoader: the multi-channel image
    and the corresponding mask.

    Args:
        dataloader (DataLoader): Source of (image, mask) batches.
        masks_dir (str): Directory of masks to infer number of classes.
        num_show (int): Number of samples (batches) to visualize.
        clip_percent (float): Used in auto-rescaling each channel.
    """
    num_classes = get_num_classes(masks_dir)

    data_iter = iter(dataloader)
    for _ in range(num_show):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        images = batch["image"]   # (B, C, H, W)
        masks = batch["mask"]    # (B, H, W)
        filenames = batch.get("filename", ["unknown"] * len(images))

        # Only show the first item in the batch
        img0 = images[0].cpu().numpy()  # (C, H, W)
        msk0 = masks[0].cpu().numpy()   # (H, W)
        fn = filenames[0]

        print(f"Displaying {img0.shape[0]}-channel image for: {fn}")
        plot_nchannel_image(img0, clip_percent=clip_percent)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(msk0, cmap="jet", vmin=0, vmax=num_classes - 1)
        ax.set_title(f"Segmentation Mask: {fn}")
        ax.axis("off")
        plt.show()


def visualize_predictions(model: nn.Module,
                         val_loader: DataLoader,
                         num_show: int = 5) -> None:
    """
    Display side-by-side: [Input image], [Ground Truth mask], and [Predicted mask].
    Uses the first 3 channels of the input image for visualization.

    Args:
        model (nn.Module): A trained segmentation model.
        val_loader (DataLoader): Dataloader for the validation dataset.
        num_show (int): How many samples to visualize.
    """
    model.eval()

    # Infer num_classes from the model if possible
    if (hasattr(model, 'segmentation_head')
        and hasattr(model.segmentation_head, 'out_channels')):
        num_classes = model.segmentation_head.out_channels
    else:
        # Fallback: guess from one batch
        first_batch = next(iter(val_loader))
        images = first_batch["image"].to(device)
        with torch.no_grad():
            logits = model(images)
        num_classes = logits.shape[1]

    fig, axes = plt.subplots(nrows=num_show, ncols=3, figsize=(15, 5 * num_show))
    val_iter = iter(val_loader)

    for i in range(num_show):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        images = batch["image"].to(device)  # (B, C, H, W)
        masks = batch["mask"].to(device)    # (B, H, W)
        fnames = batch["filename"]

        with torch.no_grad():
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

        # Only visualize the first sample in this batch
        img_np = images[0].cpu().numpy()  # (C, H, W)
        mask_gt = masks[0].cpu().numpy()  # (H, W)
        mask_pd = preds[0].cpu().numpy()  # (H, W)

        # Transpose for plotting: (H, W, C)
        img_np = np.transpose(img_np, (1, 2, 0))

        ax_orig = axes[i, 0] if num_show > 1 else axes[0]
        ax_gt = axes[i, 1] if num_show > 1 else axes[1]
        ax_pred = axes[i, 2] if num_show > 1 else axes[2]

        # Show only the first 3 channels
        ax_orig.imshow(img_np[:, :, :3])
        ax_orig.set_title(f"Image\n{fnames[0]}")
        ax_orig.axis("off")

        ax_gt.imshow(mask_gt, cmap="jet", vmin=0, vmax=num_classes - 1)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")

        ax_pred.imshow(mask_pd, cmap="jet", vmin=0, vmax=num_classes - 1)
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# 6) A One-Stop Workflow Function
# ------------------------------------------------------------------------------

def train_segmentation_workflow(images_dir: str,
                               masks_dir: str,
                               train_ratio: float = 0.8,
                               target_min: int = 200,
                               target_max: int = 400,
                               down_ratio: float = 0.25,
                               num_epochs: int = 50,
                               learning_rate: float = 1e-4,
                               load_model: bool = False,
                               model_path: str = None,
                               save_checkpoint_path: str = None) -> nn.Module:
    """
    A high-level convenience function that:
      1) Reads image metadata (#channels, height, width).
      2) Infers number of classes from mask data.
      3) Chooses an augmentation multiplier to reach a target dataset size.
      4) Splits data into train/validation subsets.
      5) Builds a UNet (ResNet34) and trains it, optionally saving checkpoints.

    Args:
        images_dir (str): Directory of multi-channel TIF images.
        masks_dir (str):  Directory of integer-labeled masks 'MASK_<filename>'.
        train_ratio (float): Fraction of data for training (0.8 by default).
        target_min (int): Minimum dataset size after augmentation.
        target_max (int): Maximum threshold above which no augmentation is added.
        down_ratio (float): Fraction for downsampling (e.g. 0.25).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for AdamW.
        load_model (bool): If True, load weights from 'model_path' before training.
        model_path (str): Path to a .pth file with saved weights.
        save_checkpoint_path (str): Where to save the model after each epoch (optional).

    Returns:
        nn.Module: The trained segmentation model.
    """
    # 1) Get image dims/channels
    orig_height, orig_width, min_channels = get_image_metadata(images_dir)

    # 2) Get number of classes
    num_classes = get_num_classes(masks_dir)

    # 3) Compute downsampled dims
    down_height, down_width = get_downsampled_dims(orig_height, orig_width, down_ratio)

    # 4) Build Albumentations pipelines
    train_aug, val_aug = get_augmentation_pipelines(down_height, down_width)

    # 5) Count base images
    image_paths = glob.glob(os.path.join(images_dir, "*.tif"))
    num_images = len(image_paths)

    # 6) Compute augmentation multiplier
    aug_mult = compute_augment_multiplier(num_images, target_min=target_min, target_max=target_max)

    # 7) Build the dataset (initially no pipeline assigned)
    full_dataset = ImageDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        aug_pipeline=None,
        augment_multiplier=aug_mult,
        num_channels=min_channels
    )

    # Split into train/val sets
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=train_ratio)

    # Because random_split returns Subset objects, we re-instantiate two new
    # datasets with the appropriate augmentation pipelines and restricted indices.
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    train_dataset = ImageDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        aug_pipeline=train_aug,
        augment_multiplier=aug_mult,
        num_channels=min_channels
    )
    train_dataset.expanded_image_paths = [train_dataset.expanded_image_paths[i] for i in train_indices]

    val_dataset = ImageDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        aug_pipeline=val_aug,
        augment_multiplier=aug_mult,
        num_channels=min_channels
    )
    val_dataset.expanded_image_paths = [val_dataset.expanded_image_paths[i] for i in val_indices]

    # 8) Build DataLoaders
    batch_size = determine_batch_size()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 9) Build (or load) the model
    model = build_model(
        device=device,
        num_classes=num_classes,
        num_channels=min_channels,
        load_model=load_model,
        model_path=model_path
    )

    # 10) Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_checkpoint_path=save_checkpoint_path
    )

    return model


# ------------------------------------------------------------------------------
# 7) Generating New ROIs on Unseen Images
# ------------------------------------------------------------------------------

def filter_largest_region_per_label(mask: np.ndarray,
                                    num_classes: int,
                                    max_components: int = 1) -> np.ndarray:
    """
    For each class label [1..(num_classes-1)], keep up to 'max_components' largest 
    connected regions. Discard smaller components.

    Args:
        mask (np.ndarray): 2D integer array of labels (0..N).
        num_classes (int): Number of classes, including background=0.
        max_components (int): Number of largest components to keep for each label.

    Returns:
        np.ndarray: A new mask of the same shape, with small components removed.
    """
    filtered = np.zeros_like(mask, dtype=mask.dtype)
    for lbl in range(1, num_classes):  # skip background=0
        binary = (mask == lbl).astype(np.uint8)
        labeled = label(binary)
        props = regionprops(labeled)
        # Sort by descending area
        props_sorted = sorted(props, key=lambda r: r.area, reverse=True)
        for region in props_sorted[:max_components]:
            filtered[labeled == region.label] = lbl
    return filtered


def create_polygons_from_mask(mask: np.ndarray,
                              approx_factor: float = 0.001) -> dict:
    """
    For each label [1..max_label], find the largest external contour,
    simplify with approxPolyDP, and return its polygon coordinates.

    Args:
        mask (np.ndarray): 2D array of integer labels.
        approx_factor (float): Factor for polygon simplification.

    Returns:
        dict: { label_val: {"x": np.array([...]), "y": np.array([...])} } 
              for each label_val > 0 found in the mask.
    """
    polygons_dict = {}
    max_label = mask.max()
    for lbl_val in range(1, max_label + 1):
        binary = (mask == lbl_val).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        # Keep the largest contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest = contours[0]

        perimeter = cv2.arcLength(largest, closed=True)
        epsilon = approx_factor * perimeter
        simplified = cv2.approxPolyDP(largest, epsilon, closed=True)

        xs = simplified[:, 0, 0]
        ys = simplified[:, 0, 1]
        polygons_dict[lbl_val] = {"x": xs, "y": ys}

    return polygons_dict


def visualize_segmentation_and_polygons(raw_image: np.ndarray,
                                        mask: np.ndarray,
                                        polygons_dict: dict,
                                        label_colors: dict = None,
                                        class_names: dict = None,
                                        alpha: int = 150,
                                        figsize: tuple = (14, 7),
                                        title_mask: str = "Segmentation Mask",
                                        title_poly: str = "Polygon Overlay") -> None:
    """
    Show side-by-side subplots:
      - Left: raw_image with a semi-transparent 'mask' overlay.
      - Right: same raw_image with polygon outlines from polygons_dict.

    Args:
        raw_image (np.ndarray): Shape (H, W, 3).
        mask (np.ndarray):      Shape (H, W), integer labels.
        polygons_dict (dict):   From create_polygons_from_mask.
        label_colors (dict):    { label_val: (R, G, B) }. If None, random colors used.
        class_names (dict):     { label_val: "Name" }. If None, default naming.
        alpha (int):            Transparency of the overlay (0..255).
        figsize (tuple):        Size of the matplotlib figure.
        title_mask (str):       Title for the left subplot.
        title_poly (str):       Title for the right subplot.
    """
    if label_colors is None:
        label_colors = {}

    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # Build semi-transparent overlay
    unique_labels = [lbl for lbl in np.unique(mask) if lbl != 0]
    for lbl_val in unique_labels:
        if lbl_val not in label_colors:
            # Random color
            color = np.random.randint(0, 256, size=3).tolist()
            label_colors[lbl_val] = tuple(color)

        region_mask = (mask == lbl_val)
        overlay[region_mask, :3] = label_colors[lbl_val]
        overlay[region_mask, 3] = alpha

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: segmentation overlay
    axes[0].imshow(raw_image)
    axes[0].imshow(overlay)
    axes[0].set_title(title_mask)
    axes[0].axis("off")

    # Right: polygon overlay
    axes[1].imshow(raw_image)

    for lbl_val, coords in polygons_dict.items():
        xs = coords["x"]
        ys = coords["y"]
        xs_closed = np.append(xs, xs[0])
        ys_closed = np.append(ys, ys[0])

        color_array = np.array(label_colors[lbl_val]) / 255.0
        lab_name = f"Label {lbl_val}"
        if class_names and lbl_val in class_names:
            lab_name = class_names[lbl_val]

        axes[1].plot(xs_closed, ys_closed, marker="o", markersize=4,
                     linestyle="-", linewidth=2, color=color_array, label=lab_name)

    axes[1].set_title(title_poly)
    axes[1].legend()
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def save_mask_as_fiji_rois(mask: np.ndarray,
                           base_name: str,
                           output_dir: str,
                           region_names=None,
                           approx_factor: float = 0.001,
                           keep_background: bool = False) -> None:
    """
    Convert each labeled region in 'mask' into a Fiji-compatible .roi,
    and write them into a single .zip named "<base_name> ROI.zip".

    Args:
        mask (np.ndarray): 2D array (H, W), integer-labeled, 0 for background.
        base_name (str):   Filename prefix for the .zip.
        output_dir (str):  Where to save the "<base_name> ROI.zip".
        region_names (dict or list): 
            - dict: { label_val: "Name" }
            - list: e.g. region_names[i-1] is "Name" for label=i
            - None -> use "Label1", "Label2", etc.
        approx_factor (float): Factor for cv2.approxPolyDP to simplify contours.
        keep_background (bool): If True, also creates a ROI for label=0.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{base_name} ROI.zip")

    labels_in_mask = np.unique(mask)
    if not keep_background:
        labels_in_mask = labels_in_mask[labels_in_mask != 0]

    # If region_names is a list, convert to a dict
    if isinstance(region_names, list):
        region_names_dict = {}
        for i, name in enumerate(region_names, start=1):
            region_names_dict[i] = name
        region_names = region_names_dict
    elif region_names is None:
        region_names = {}

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for lbl_val in labels_in_mask:
            lbl_name = region_names.get(lbl_val, f"Label{lbl_val}")

            binary = (mask == lbl_val).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            # Keep largest contour
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]

            perimeter = cv2.arcLength(largest_contour, closed=True)
            epsilon = approx_factor * perimeter
            simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

            xs = simplified[:, 0, 0]
            ys = simplified[:, 0, 1]

            roi = roifile.ImagejRoi.frompoints(list(zip(xs, ys)))
            roi.roitype = roifile.ROI_TYPE.POLYGON

            # Write each ROI as "<LabelName>.roi"
            zipf.writestr(f"{lbl_name}.roi", roi.tobytes())

    print(f"Saved Fiji ROIs to: {zip_path}")


def predict_unseen_images(unseen_images_dir: str,
                          masks_dir: str,
                          output_masks_dir: str,
                          output_rois_dir: str,
                          model_path: str,
                          down_ratio: float = 0.25,
                          keep_largest: bool = True,
                          max_components: int = 1,
                          visualize: bool = True,
                          side_by_side_plot: bool = True,
                          label_colors: dict = None,
                          class_names: dict = None,
                          approx_factor: float = 0.001) -> None:
    """
    Run inference on unseen images using a trained model.

    Steps:
        1) Read shape/channels from unseen images.
        2) Infer number of classes from 'masks_dir'.
        3) Load the saved model checkpoint.
        4) Resize images for inference, forward pass, upsample mask.
        5) (Optional) Keep only largest connected components per label.
        6) Save final mask as "MASK_<base_name>.tif".
        7) Save polygons in a ".zip" of Fiji .roi files.
        8) (Optional) Visualize side-by-side segmentation overlay & polygon overlay.

    Args:
        unseen_images_dir (str): Directory with new images (TIF, PNG, JPG).
        masks_dir (str): Used to determine 'num_classes' for the model.
        output_masks_dir (str): Where to save predicted masks as TIF.
        output_rois_dir (str):  Where to save ROI zip files.
        model_path (str): Path to the .pth model checkpoint.
        down_ratio (float): Same ratio used during training for resizing.
        keep_largest (bool): If True, keep only the largest region(s) per label.
        max_components (int): Number of largest components to keep per label if keep_largest=True.
        visualize (bool): If True, display each predicted result in a matplotlib figure.
        side_by_side_plot (bool): If True, show segmentation overlay and polygon overlay side-by-side.
        label_colors (dict): { label_val: (R, G, B) } for coloring overlays. None -> random colors.
        class_names (dict): { label_val: "Name" } for labeling polygons. None -> default "LabelX".
        approx_factor (float): For polygon simplification in cv2.approxPolyDP.
    """
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_rois_dir, exist_ok=True)

    # 1) Infer shape and channel count
    orig_height, orig_width, in_channels = get_image_metadata(unseen_images_dir)
    print(f"Detected from unseen images: height={orig_height}, width={orig_width}, channels={in_channels}")

    # 2) Determine number of classes
    num_classes = get_num_classes(masks_dir)
    print(f"Detected number of classes = {num_classes} (including background).")

    # 3) Load the model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4) Albumentations resizing
    down_height, down_width = get_downsampled_dims(orig_height, orig_width, down_ratio)
    print(f"Inference resizing to: {down_height} x {down_width} (ratio={down_ratio})")
    inference_transform = A.Compose([
        A.Resize(height=down_height, width=down_width),
    ])

    # 5) Collect new images
    all_image_files = sorted(
        glob.glob(os.path.join(unseen_images_dir, "*.tif")) +
        glob.glob(os.path.join(unseen_images_dir, "*.png")) +
        glob.glob(os.path.join(unseen_images_dir, "*.jpg"))
    )
    if not all_image_files:
        print("No images found in unseen_images_dir.")
        return

    # 6) Inference loop
    for img_path in tqdm(all_image_files, desc="Predicting"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Load the raw image (could be (H, W), (C, H, W), or (H, W, C))
        raw = tifffile.imread(img_path)
        if raw.ndim == 2:
            # shape=(H, W). If model expects >1 channel, replicate
            if in_channels > 1:
                raw = np.stack([raw] * in_channels, axis=0)
            else:
                raw = raw[np.newaxis, ...]
        elif raw.ndim == 3:
            # Could be (C, H, W) or (H, W, C)
            if raw.shape[0] != in_channels and raw.shape[-1] == in_channels:
                # Transpose to (C, H, W)
                raw = np.transpose(raw, (2, 0, 1))

        # If we have extra channels, slice them
        if raw.shape[0] > in_channels:
            raw = raw[:in_channels, :, :]

        # If fewer channels, replicate the last one
        while raw.shape[0] < in_channels:
            raw = np.concatenate([raw, raw[-1:]], axis=0)

        # Auto-rescale each channel
        raw_rescaled = np.stack([auto_rescale(ch) for ch in raw], axis=0)

        # Albumentations expects (H, W, C)
        resized = inference_transform(image=raw_rescaled.transpose(1, 2, 0))["image"]

        # Convert to Torch tensor (C, H, W) and scale to 0..1
        img_tensor = torch.tensor(resized.transpose(2, 0, 1), dtype=torch.float32, device=device).unsqueeze(0)/255.0

        # Forward pass
        with torch.no_grad():
            logits = model(img_tensor)
            pred_mask_small = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Upsample to original size
        pred_mask = cv2.resize(pred_mask_small, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

        # Keep largest components if requested
        if keep_largest:
            pred_mask = filter_largest_region_per_label(pred_mask, num_classes, max_components=max_components)

        # Save predicted mask
        out_mask_path = os.path.join(output_masks_dir, f"MASK_{base_name}.tif")
        cv2.imwrite(out_mask_path, pred_mask.astype(np.uint8))

        # Save Fiji ROIs
        save_mask_as_fiji_rois(
            mask=pred_mask,
            base_name=base_name,
            output_dir=output_rois_dir,
            region_names=class_names,
            approx_factor=approx_factor
        )

        # Visualization
        if visualize and side_by_side_plot:
            c = raw.shape[0]
            if c >= 3:
                rgb = raw[:3].transpose(1, 2, 0)
            else:
                # Expand to 3 channels
                replicate_count = 3 - c
                extended = np.concatenate([raw, np.tile(raw[-1:], (replicate_count, 1, 1))], axis=0)
                rgb = extended[:3].transpose(1, 2, 0)

            rgb_vis = auto_rescale(rgb).astype(np.float32) / 255.0
            rgb_vis = cv2.resize(rgb_vis, (orig_width, orig_height))

            polygons_dict = create_polygons_from_mask(pred_mask, approx_factor=approx_factor)

            visualize_segmentation_and_polygons(
                raw_image=rgb_vis,
                mask=pred_mask,
                polygons_dict=polygons_dict,
                label_colors=label_colors,
                class_names=class_names,
                alpha=150,
                figsize=(14, 7),
                title_mask=f"Segmentation: {base_name}",
                title_poly=f"Polygon Overlay: {base_name}"
            )

    print("Finished inference for all images.")
