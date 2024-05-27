import os
import json
import random
import string
from pathlib import Path
from datetime import datetime

def generate_unique_string(length=6):
    """Generate a unique random string of specified length."""
    chars = string.ascii_lowercase
    return datetime.now().strftime("%m-%d-%H-%M-%S-") + ''.join(random.choice(chars) for _ in range(length)) 

def create_unique_folder(model_name, train_file, base_folder):
    """Create a unique folder within the base folder."""
    # Generate unique string for folder name
    unique_string = generate_unique_string()

    # Create folder with the unique name
    folder_path = os.path.join(base_folder, model_name, Path(train_file).stem, unique_string)
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

def create_model_folder(model_name, train_file, args, params_set, base_folder='models'):
    """Copy corresponding inner dict to a new JSON file in the new folder."""

    new_folder_path = create_unique_folder(model_name, train_file, base_folder)

    params_file_path = os.path.join(new_folder_path, "params.json")
    with open(params_file_path, 'w') as f:
        json.dump(params_set, f, indent=4)
    args_file_path = os.path.join(new_folder_path, "args.json")
    with open(args_file_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    return new_folder_path