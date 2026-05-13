Macroinvertebrate EDA Program README

== PROJECT GOAL ==
This application takes the Kaggle macroinvertebrate dataset, performs exploratory data analysis (EDA), and presents the results in an accessible, layperson-friendly manner using a Tkinter GUI. It provides both basic statistical summaries and graphical outputs about the image dataset.

== MAIN FEATURES ==
- Dataset indexing: reads raw image data and formats it into a structured DataFrame.
- Class distribution analysis: bar chart showing the number of images per macroinvertebrate class.
- Image size analysis: histogram of image width and height distributions.
- Image grid generation: random per-class grid (17 images) and random sample grid (9 images).
- Dataset summary: total image count, class count, and mean image dimensions.
- Tkinter GUI: simple, button-driven interface for running all EDA outputs.

== PACKAGES USED ==
- pandas
- opencv-python
- matplotlib
- seaborn
- Pillow
- tkinter
- pathlib
- sys

== INSTALLATION ==
1. Clone or download this repository: https://github.com/jaib2203/st1-assignment3-group1
2. Install relevant dependencies: pip install -r requirements.txt
3. Place your dataset inside the data/raw directory. It should be organised into subdirectories per class.

Note: this program must be launched from an IDE that supports opening a project folder (e.g. VSCode), not by running main.py in isolation. This is due to filepaths.

== HOW TO RUN ==
1. Launch main.py and run the script.

== AUTHORS ==
Ajugo Maxwel Vuni - Dataset Indexer & GUI implementation
Jai Butler - EDA Service & program integration