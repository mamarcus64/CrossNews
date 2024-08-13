# CrossNews
Data and experimental code for the CrossNews authorship dataset.
## Conda Environment Installation
Please run the following code to generate and activate the CrossNews conda environment:

```
conda create --name CrossNews --file requirements.txt`
conda activate CrossNews
```

## Dataset Extraction
The gold and silver splits of CrossNews are located in `raw_data.zip`. Extract this file into the `raw_data` folder. To generate the dataframes used in experimental training, please run:

`python src/dataset_creation.py`

## Usage

See `src/main.py` for command line details. Example usage:

```
python src/main.py \
--model random \
--train \
--train_file verification_data/train/CrossNews_Article_Article.csv \
--parameter_sets all \
--test \
--test_files verification_data/train/CrossNews_Article_Article.csv \
verification_data/train/CrossNews_Article_Tweet.csv \
verification_data/train/CrossNews_Tweet_Tweet.csv
```
