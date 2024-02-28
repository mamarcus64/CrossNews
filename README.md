# CrossNews
Data for the CrossNews authorship dataset.
## Usage
The gold and silver splits of CrossNews are located in `data.zip`. To generate the dataframes used in experimental training, please see `ids_to_files.py`. For example, to generate the dataframes for the main experiment, extract `data.zip` into the `data` folder, then run:

`python ids_to_files.py --id_file main_experiment`
