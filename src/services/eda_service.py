# eda_service.py
# Generates the analytical & visual data for the dataset.
""" Much of the code used below was adapted from the assignment guidance document."""

import sys
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1])) # Required to resolve path to config.py
from config import EDA_OUTPUT_DIR, IMAGE_SIZE

class EDAService:
    def __init__(self, dataframe: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR):
        self.dataframe = dataframe
        self.output_dir = output_dir

    # Generates and saves a bar chart showing the number of images per macroinvertebrate class.
    def generate_class_count(self) -> None:
        self.dataframe["label"].value_counts().plot(kind="bar")
        plt.title("Image Per Macroinvertebrate Class")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_count.jpg")
        plt.close()

    # Generates and saves a histogram showing the distribution of image widths/heights.
    def generate_image_size_distribution(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(self.dataframe["width"], bins=20, ax=axes[0])
        sns.histplot(self.dataframe["height"], bins=20, ax=axes[1])
        axes[0].set_title("Image Width Distribution")
        axes[1].set_title("Image Height Distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / "image_size_distribution.jpg")
        plt.close()
    
    # Generates and saves a grid of 9 sample images, selected randomly from the dataset each time. 
    def generate_sample_grid(self) -> None:
        samples = self.dataframe.sample(9)

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        for ax, (_, row) in zip(axes.flat, samples.iterrows()):
            image = cv2.imread(row["file_path"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            ax.imshow(image)
            ax.set_title(row["label"])
            ax.axis("off")  # hides the x/y axis ticks
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_images.png")
        plt.close()

    # Generates and prints a summary of dataset data.
    def generate_summary(self) -> None:
        total_images = len(self.dataframe)
        total_classes = self.dataframe["label"].nunique()
        mean_width = self.dataframe["width"].mean()
        mean_height = self.dataframe["height"].mean()
        print("=== DATASET SUMMARY ===")
        print()
        print(f"Total Images: {total_images}")
        print(f"Total Classes: {total_classes}")
        print(f"Mean Image Width: {mean_width:.2f}")
        print(f"Mean Image Height: {mean_height:.2f}")