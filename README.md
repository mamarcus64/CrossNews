# CrossNews
Data and experimental code for the CrossNews authorship dataset.
## Conda Environment Installation
Please run the following code to generate and activate the CrossNews conda environment:

```
conda create --name CrossNews --file requirements.txt`
conda activate CrossNews
```

Additonal Conda environments for the PART, STEL, and LUAR environments are available in the `conda_environments` folder.

## Dataset Extraction
The gold and silver splits of CrossNews are located in `raw_data.zip`. Extract this file into the `raw_data` folder. To generate the dataframes used in experimental training, please run:

`python src/dataset_creation.py`

## Usage

Run `src/run_verification.py` and `src_run_attribution.py` accordingly.

Example verification usage:

```
python src/run_verification.py \
--model ngram \
--train \
--train_file verification_data/train/CrossNews_Article_Article.csv \
--parameter_sets default \
--save_folder results \
--test \
--test_files verification_data/test/CrossNews_Article_Article.csv
```

Example attribution usage:

```
python src/run_attribution.py \
--model ngram_aa \
--train \
--query_file attribution_data/query/CrossNews_Article.csv \
--parameter_sets default \
--test \
--target_file attribution_data/test/CrossNews_Tweet.csv
```

To train embedding models, see `src/train_embedding.py`. To use SELMA, the embeddings must be generated first. See `src/generate_selma_embeddings.py` for more details.
