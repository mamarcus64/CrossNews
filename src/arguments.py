import argparse
import os

def check_file_path(file_path):
    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError(f"File path {file_path} does not exist.")
    return file_path

def get_parser():
    parser = argparse.ArgumentParser(description="Command line interface for model training and testing.")

    # Required arguments
    parser.add_argument("--model", type=str, help="Name of the model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--load", action="store_true", help="Load the model from saved folder")
    parser.add_argument("--load_folder", type=str, help="Model folder, if loading")

    # Optional arguments
    parser.add_argument("--save_folder", default="models", help="Save folder location")
    parser.add_argument("--train_file", type=check_file_path, help="File path for training")
    parser.add_argument("--parameter_set", type=str, help="Parameter set for training.", default="default")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="Proportion of train data to be reserved for eval")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--test_files", type=check_file_path, nargs='+', help="File path(s) for testing")

    args = parser.parse_args()

    # Validate model version if provided
    if args.load_folder:
        check_file_path(args.load_folder)

    # Additional validation based on train or test flags
    if args.train:
        if not args.train_file:
            parser.error("--train_file is required when --train is set.")
        if not args.parameter_set:
            parser.error("--parameter_set is required when --train is set.")

    if args.test and not args.test_files:
        parser.error("--test_file is required when --test is set.")

    return args