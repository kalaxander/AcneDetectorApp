import os

# This gets the path of the 'src' directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# This gets the path of the main project folder
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Now, we define the paths to your files
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'acne_transfer_model_finetuned.h5')
LABELS_PATH = os.path.join(PROJECT_ROOT, 'labels.txt')