import argparse
import os

def check_file_path(file_path):
    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError(f"File path {file_path} does not exist.")
    return file_path

def check_evaluation_criteria(criteria):
    valid_criteria = ["F1", "Accuracy", "AUC"]
    if criteria not in valid_criteria:
        raise argparse.ArgumentTypeError(f"Evaluation criteria must be one of {', '.join(valid_criteria)}")
    return criteria


def get_parser():
    parser = argparse.ArgumentParser(description="Command line interface for model training and testing.")

    # Required arguments
    parser.add_argument("--model", type=str, help="Name of the model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--load", action="store_true", help="Load the model from saved folder")
    parser.add_argument("--load_folder", type=str, help="Model folder, if loading")

    # Optional arguments
    parser.add_argument("--train_file", type=check_file_path, help="File path for training")
    parser.add_argument("--parameter_sets", type=str, nargs='+', help="Parameter set(s) for training. If training all, set to \"all\". "
                    + "Each parameter set will be tested on the evaluation data and the best parameter set will be tested.", default=["default"])
    parser.add_argument("--evaluation_metric", type=check_evaluation_criteria, help="Evaluation criteria (F1, Accuracy, AUC)", default="F1")
    parser.add_argument("--keep_all_models", action="store_true", help="Whether to keep all of the parameter sets vs. just the one that performs "
                        + "the best on the evaluation set. Default: False")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for train/eval splitting")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="Proportion of train data to be reserved for eval")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--test_files", type=check_file_path, nargs='+', help="File path(s) for testing")
    parser.add_argument("--silent", action="store_true") 

    args = parser.parse_args()

    # Validate model version if provided
    if args.load_folder:
        check_file_path(args.load_folder)

    # Additional validation based on train or test flags
    if args.train:
        if not args.train_file:
            parser.error("--train_file is required when --train is set.")
        if not args.parameter_sets:
            parser.error("--parameter_set is required when --train is set.")

    if args.test and not args.test_files:
        parser.error("--test_file is required when --test is set.")

    return args