# BrainSeg2D

**BrainSeg2D** is a Python library for segmenting 2D brain slice images using [PyTorch](https://pytorch.org/) and [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch), with labeled ROI support for [Fiji (ImageJ)](https://imagej.net/software/fiji/). It supports multi-channel TIFF images, applies automatic data augmentations (via [albumentations](https://albumentations.ai/)), trains a UNet model, and generates ImageJ/Fiji ROI files for each predicted label. The library also includes a Fiji macro to convert initial ROIs (polygons) into labeled masks for model training.

---

## Features

- **Multi-Channel TIFF Support**: Automatically detects the number of channels and includes them in training/inference.
- **Flexible Augmentations**: Automatically adjusts and augments the dataset with flipping, rotation, resizing, and other transformations—especially useful when labeled data is limited or images have very high resolution.
- **End-to-End Training Workflow**: `train_segmentation_workflow(...)` handles dataset creation, augmentation, and model training.
- **ROI File Generation**: Exports segmentation masks as `.roi` files in a `.zip` archive, compatible with Fiji/ImageJ.

---

## Project Status

🚧 **BrainSeg2D is under active development.** Features may change, and improvements are ongoing. Feedback and contributions are welcome!

---

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/YOUR_USERNAME/brainseg2d.git
cd brainseg2d
```

### 2. Create and Activate a Python Environment

#### Option A: Using Conda

```sh
conda env create -f environment.yml
conda activate brainseg2d
```

#### Option B: Using venv + Pip

```sh
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 3. (Optional) Install BrainSeg2D in Editable Mode 

```sh
pip install -e .
```

This allows you to `import brainseg2d` from anywhere on your system.

### 4. Additional Requirements for Fiji Integration

To use the Fiji macro script for converting ROI zips into labeled masks, ensure you have [ImageJ/Fiji](https://imagej.net/software/fiji/) installed.

---

## Usage

### 1. Generate Labeled Masks in Fiji

To convert ROI files into labeled masks, use the Fiji macro script included in `fiji_macro/generate_masks_from_rois.ijm`.

#### Steps:

1. Open Fiji.
2. Go to **File > Open...** and select `generate_masks_from_rois.ijm`.
3. Click **Run**.
4. When prompted, select:
   - The **input folder** containing both TIFF images and corresponding ROI zip files (e.g., the `Data` placeholder folder).
   - The **output folder** where the labeled masks will be saved as `MASK_<filename>.tif` (e.g., the `Masks` placeholder folder).

#### Example:

| Input Image  | ROI File     | Output Labeled Mask |
| ------------ | ------------ | ------------------- |
| `img001.tif` | `img001.zip` | `MASK_img001.tif`   |
| `img002.tif` | `img002.zip` | `MASK_img002.tif`   |

---

### 2. Train a Model

```python
from brainseg2d.brainseg2d import train_segmentation_workflow

model = train_segmentation_workflow(
    images_dir="path/to/train_images",
    masks_dir="path/to/train_masks",
    train_ratio=0.8,    # 80% training, 20% validation
    down_ratio=0.25,    # Downsample input images by 25%
    num_epochs=50,
    save_checkpoint_path="./model_checkpoints/unet_resnet34.pth"
)
```

#### Input Data Format:

- ``: Folder containing TIFF images, each shaped `(Channels, Height, Width)`.
- ``: Corresponding segmentation masks, with filenames prefixed by `MASK_` (e.g., `MASK_img001.tif`).

#### Output:

- A trained UNet model saved at `save_checkpoint_path`.
- A log of training progress, including loss and accuracy metrics (e.g., % pixel accuracy).

---

### 3. Perform Inference on New Images

```python
from brainseg2d.brainseg2d import predict_unseen_images

predict_unseen_images(
    unseen_images_dir="path/to/new_images",
    masks_dir="path/to/train_masks",  # Used to determine the number of classes
    output_masks_dir="./Predicted_Masks",
    output_rois_dir="./Predicted_ROIs",
    model_path="./model_checkpoints/unet_resnet34.pth",
    visualize=True,       # Displays overlay plots
    side_by_side_plot=True
)
```

#### What This Does:

1. Loads the trained model checkpoint.
2. Processes each `.tif` file in `unseen_images_dir`.
3. Predicts segmentation masks for each image.
4. Exports ROI files (`.zip` format) for ImageJ/Fiji.
5. Optionally visualizes the results side by side using Matplotlib.

---

### 4. Explore the Example Notebooks (In Progress...)

Use the provided notebooks as templates to customize your own segmentation pipeline. They allow you to fine-tune and control every step of the training and inference process.

---

## Contributing

I would certainly welcome contributions! To contribute:

1. **Fork** the repository.
2. **Create a branch** for your feature or bug fix.
3. **Submit a pull request** after testing your changes.

For bug reports or feature requests, please [open an issue](https://github.com/adriamilcar/brainseg2d/issues).

---

## Citing BrainSeg2D

If you use **BrainSeg2D** in your research, please cite it using the following format:

**BibTeX citation:**
```bibtex
@misc{brainseg2d,
  author = {Adrián F. Amil},
  title = {BrainSeg2D},
  year = {2025},
  howpublished = {\url{https://github.com/adriamilcar/brainseg2d}},
}

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](https://mit-license.org/) file for details.

© 2025 **Adrián Fernández Amil** – Please credit the original author if you modify or distribute this work.

---
