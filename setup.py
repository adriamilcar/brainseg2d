# setup.py
from setuptools import setup, find_packages

setup(
    name="brainseg2d",
    version="0.1.0",
    description="BrainSeg2D",
    author="Adrián F. Amil",
    packages=find_packages(),
    install_requires=[
        "torch",
        "segmentation-models-pytorch",
        "albumentations",
        "tifffile",
        "opencv-python",
        "scikit-image",
        "roifile",
        "tqdm",
        "matplotlib",
        "numpy",
        "glob2",
        "torchvision"
    ],
    python_requires=">=3.7",
)
