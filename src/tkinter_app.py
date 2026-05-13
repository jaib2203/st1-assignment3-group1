# tkinter_app.py
# Creates an app that will host and display the output from the dataset & EDA model.

# Import Tkinter module that is used to build the app
import tkinter as tk
from PIL import Image, ImageTk
import sys
from pathlib import Path 

"""Create a path to access the dataset_indexer, eda_service, and config codes"""
# Required to resolve path to workflow_service.py
sys.path.insert(0, str(Path(__file__).parent / "services")) 
from workflow_service import WorkflowService

# Create a class that holds the dataset and it's output during execution.
class GUI_App:
    def __init__(self, workflow: WorkflowService):
        self.workflow = workflow
        self.MainWindow = tk.Tk()
        self.MainWindow.title("Dataset Indexer and EDA Service")
        self.MainWindow.geometry("900x700")

        # Configure grid weights so the result area expands with the window.
        self.MainWindow.columnconfigure(1, weight=1)
        self.MainWindow.rowconfigure(1, weight=1)

        # Title label spanning full width.
        self.title_label = tk.Label(
            self.MainWindow,
            text="A Dataset Indexer and EDA Service Application",
            font=('Aptos Display', 16)
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=15)

        # Left sidebar frame for buttons.
        self.leftside_frame = tk.Frame(self.MainWindow, padx=10)
        self.leftside_frame.grid(row=1, column=0, sticky="ns")

        # Result area frame.
        self.resultbox_frame = tk.Frame(self.MainWindow, bg="light green")
        self.resultbox_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        # Buttons.
        btn_home = tk.Button(self.leftside_frame, text="Home", width=20, command=self.welcome_message) # Home message label
        btn_home.pack(pady=5, anchor="w")
        btn_summary = tk.Button(self.leftside_frame, text="Load Dataset Summary", width=20, command=self.dataset_summary)
        btn_summary.pack(pady=5, anchor="w")
        btn_class_count = tk.Button(self.leftside_frame, text="Class Count", width=20, command=self.class_count)
        btn_class_count.pack(pady=5, anchor="w")
        btn_image_distribution = tk.Button(self.leftside_frame, text="Image Size Distribution", width=20, command=self.image_distribution)
        btn_image_distribution.pack(pady=5, anchor="w")
        btn_one_per_grid = tk.Button(self.leftside_frame, text="An Image Per Class Grid", width=20, command=self.image_per_class)
        btn_one_per_grid.pack(pady=5, anchor="w")
        btn_sample_grid = tk.Button(self.leftside_frame, text="Sample Images", width=20, command=self.sample_image)
        btn_sample_grid.pack(pady=5, anchor="w")
        btn_exit = tk.Button(self.leftside_frame, text="Exit Program", width=20, command=self.MainWindow.destroy)
        btn_exit.pack(pady=(20, 5), anchor="w")

        # Result label inside the result frame.
        self.result_label = tk.Label(self.resultbox_frame, bg="light green", anchor="center")
        self.result_label.pack(expand=True, fill="both")

        tk.mainloop()

    def show_chart(self, chart_path: Path) -> None:
        """Display a chart image in the result area."""
        image = Image.open(chart_path)
        image.thumbnail((750, 550))
        photo = ImageTk.PhotoImage(image)
        self.result_label.configure(image=photo, text="")
        self.result_label.image = photo

    """Create a welcome message def method."""
    def welcome_message(self):
        self.result_label.configure(text="=================================================================\n"
                
                "Hi Everyone!\n"
                "=================================================================\n"               
                "Welcome to the Dataset Indexer and EDA Service Application\n"
                "build by Max and Jai! We are honored to see you here!\n"
                "=================================================================\n"
                "This app has some cool features that you can interact with\n"
                "i.e., \n"
                "1. The 'Load Dataset Summary' to see the summary of the dataset.\n"
                "2. Class Count will generate a bar chart showing a bar chart of all images \n"
                "in a class,\n"
                "3. image size distribution shows the distribution of image widths/heights\n"
                "across the classes, and so on!\n"
                "=================================================================\n"
                "To know more, why don't you just click on the buttons to your left and\n"
                "see what this app can do!\n"
                "=================================================================\n"
                "Thanks for stopping by.\n"
                "=================================================================\n",
                font=("Arial", 10), justify="left",)
    
    """Create a method to hold the function for each of the buttons here"""
    def dataset_summary(self):
        summary = self.workflow.generate_summary()
        self.result_label.configure(text=summary, image="", font=("Courier", 11), justify="left")

    def class_count(self):
        chart_path = self.workflow.generate_eda_output("class_count")
        self.show_chart(chart_path)

    def image_distribution(self):
        chart_path = self.workflow.generate_eda_output("image_size_distribution")
        self.show_chart(chart_path)
        
    def image_per_class(self):
        chart_path = self.workflow.generate_eda_output("one_per_class_grid")
        self.show_chart(chart_path)
    
    def sample_image(self):
        chart_path = self.workflow.generate_eda_output("sample_images")
        self.show_chart(chart_path)
