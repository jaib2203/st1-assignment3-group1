# eda_service.py
# Generates the analytical & visual data for the dataset.
# A lot of the code used below was copied/adapted from the assignment guidance document.

import sys
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import dataset_indexer

sys.path.insert(0, str(Path(__file__).parents[1])) # Required to resolve path to config.py
from config import EDA_OUTPUT_DIR, IMAGE_SIZE

class EDAService:
    def __init__(self, dataframe: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR):
        self.dataframe = dataframe
        self.output_dir = output_dir

    def generate_class_count(self) -> None:
        """Generates and saves a bar chart showing the number of images per macroinvertebrate class."""
        self.dataframe["label"].value_counts().plot(kind="bar")
        plt.title("Image Per Macroinvertebrate Class")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_count.jpg")
        plt.close()

    def generate_image_size_distribution(self) -> None:
        """Generates and saves a histogram showing the distribution of image widths/heights."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(self.dataframe["width"], bins=20, ax=axes[0])
        sns.histplot(self.dataframe["height"], bins=20, ax=axes[1])
        axes[0].set_title("Image Width Distribution")
        axes[1].set_title("Image Height Distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / "image_size_distribution.jpg")
        plt.close()
     
    def generate_sample_grid(self) -> None:
        """Generates and saves a grid of 9 sample images, selected randomly from the dataset each time."""
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

    def generate_image_per_class_grid(self) -> None:
        """As an alternative to the random grid generator, this class generates a grid of images with
        one image from each class."""
        classes = self.dataframe["label"].unique()
        cols = 4
        rows = -(-len(classes) // cols)  # ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        for ax, label in zip(axes.flat, classes):
            row = self.dataframe[self.dataframe["label"] == label].iloc[0]
            image = cv2.imread(row["file_path"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_title(label, fontsize=9)
            ax.axis("off")
        for ax in axes.flat[len(classes):]:
            ax.axis("off")
        plt.suptitle("One Sample Per Class")
        plt.tight_layout()
        plt.savefig(self.output_dir / "one_per_class_grid.png")
        plt.close()
        

    def generate_summary(self) -> dict:
        """Generates and returns a summary of dataset data."""
        return {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width": float(self.dataframe["width"].mean()),
            "mean_height": float(self.dataframe["height"].mean()),
        }