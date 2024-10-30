# CrossNews
Data and experimental code for the CrossNews authorship dataset, presented in the paper `CROSSNEWS: A Cross-Genre Authorship Verification and Attribution Benchamrk` (link to be added). The gold and silver data of CrossNews is present in the `raw_data` zip file, with each document in the relevant json file having an `id`, an `author`, a `genre`, and the document `text` as well. Gold articles also have a `topic` field.


## Conda Environment Installation
Please run the following code to generate and activate the CrossNews conda environment:

```
conda env create -f conda_environments/authorship.yaml
conda activate authorship
```
The conda environments `luar.yaml`, `part.yaml`, and `stel.yaml` can also be used for each of the LUAR, PART, and STEL embedding methods in the full release of CrossNews.

## Usage

Example usage to train an N-gram verification model on the Article-Article train data and test on all testing splits:

```
python src/run_verification.py \
--model ngram \
--train \
--train_file verification_data/train/CrossNews_Article_Article.csv \
--test \
--test_files verification_data/train/CrossNews_Article_Article.csv \
verification_data/train/CrossNews_Article_Tweet.csv \
verification_data/train/CrossNews_Tweet_Tweet.csv
```

And to train and test an N-gram attribution model:

```
python src/run_attribution.py \
--model ngram_aa \
--train \
--train_file attribution_data/train/CrossNews_Article.csv \
--test \
--test_files attribution_data/test/CrossNews.csv
```

## Results
Results are stored in the `models` folder under the relevant train dataset name. These folders contain all of the information to re-load a model after training, and evaluation results are found in the file `test_results.json`.
