# workflow_service.py
# Coordinates the workflow of the various service classes & the tkinter GUI.
# Some parts of this script were adapted from the assignment guidance document.

from pathlib import Path
import pandas as pd

from dataset_indexer import DatasetIndexer
from eda_service import EDAService
from config import EDA_OUTPUT_DIR

class WorkflowService:
    def __init__(self):
        self.ds_indexer = DatasetIndexer()
        self.dataframe = self.ds_indexer.build_dataframe() # Loads the dataframe on init.
        self.eda = EDAService(self.dataframe)
        self.output_dir = EDA_OUTPUT_DIR

    def get_dataframe(self) -> pd.DataFrame | None:
        """Returns the loaded dataframe."""
        if self.dataframe is None:
            print("Error: dataframe not loaded.")
        return self.dataframe
    
    def generate_eda_output(self, chart_type:str) -> Path | None:
        """
        Generates and saves the relevant graph/grid, returning the file path.

        Parameters
        ----------
        type : string
            Specifies the image to be generated. Must be one of the following strings:
            "class_count" : class count bar graph.
            "image_size_distribution" : image width & heights distribution bar graph.
            "sample_images" : grid of 9 sample images.
            "one_per_class_grid" : grid of images, one selected at random from each image class.
        """

        # Delete existing file (if present) before generating.
        output_path = self.output_dir / f"{chart_type}.jpg"
        if output_path.exists():
            output_path.unlink()

        # Generate the corresponding EDA Service image.
        match chart_type:
            case "class_count":
                self.eda.generate_class_count(self.output_dir) 
            case "image_size_distribution":
                self.eda.generate_image_size_distribution(self.output_dir)
            case "sample_images":
                self.eda.generate_sample_grid(self.output_dir) 
            case "one_per_class_grid":
                self.eda.generate_image_per_class_grid(self.output_dir)
            case _:
                raise ValueError(f"Unknown EDA type: '{chart_type}'. Must be valid option.")
        
        return output_path

    def generate_summary(self) -> str:
        """Returns summary statistics for the loaded dataset."""
        summary = self.eda.generate_summary()
        return f"--- DATASET STATISTICAL SUMMARY --- \
                \nTotal Images: {summary["total_images"]} \
                \nTotal Classes: {summary["total_classes"]} \
                \nMean Width: {summary["mean_width"]:.2f}px \
                \nMean Height: {summary["mean_height"]:.2f}px"