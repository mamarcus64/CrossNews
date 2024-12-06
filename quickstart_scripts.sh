# salloc -c 8
# model="ngram"
# conda activate AuthorID_copy

# salloc -c 16
# model="ppm"
# conda activate AuthorID_copy

# salloc -c 8 -G a40
# model="o2d2"
# conda activate AuthorID_copy

salloc -c 16 -G a40
model="part_av"
conda activate part

# salloc -c 16 -G a40
# model="luar_av"
# conda activate luar

# salloc -c 16 -G a40
# model="stel_av"
# conda activate stel

# dataset="CrossNews_Article_Article.csv"
# dataset="CrossNews_Article_Tweet.csv"
# dataset="CrossNews_Tweet_Tweet.csv"
# dataset="CrossNews_Article_Article_short.csv"
# dataset="CrossNews_Article_Tweet_short.csv"
# dataset="empty.csv"
dataset="CrossNews_mini.csv"

# params="default"
params=${dataset}
# params="empty"

# save_folder="models"
save_folder="runs"

date
cd /nethome/mma81/storage/CrossNews
current_date=$(date +"%m_%d-%H_%M")
output_file="logs/${current_date}_${model}_${dataset}.txt"
python src/run_verification.py \
--model ${model} \
--train \
--save_folder ${save_folder} \
--train_file verification_data/train/${dataset} \
--parameter_sets ${params} \
--test \
--test_files verification_data/test/CrossNews_Tweet_Tweet.csv \
verification_data/test/CrossNews_Article_Article.csv \
verification_data/test/CrossNews_Article_Tweet.csv \
# verification_data/test/CrossNews_Article_Article_short.csv \
# verification_data/test/CrossNews_Article_Tweet_short.csv \
| tee $output_file
date
exit


python src/run_verification.py --model luar_av --load --load_folder models/luar_av/empty/06-01-06-25-36-wrfkgb/embeddings --parameter_sets empty.csv --test --test_files verification_data/test/CrossNews_Tweet_Tweet.csv verification_data/test/CrossNews_Article_Article.csv verification_data/test/CrossNews_Article_Tweet.csv verification_data/test/CrossNews_Article_Article_short.csv verification_data/test/CrossNews_Article_Tweet_short.csv


# salloc -c 16 -G a40
# model="part"
# conda activate part

# salloc -c 16 -G a40
# model="luar"
# conda activate luar

salloc -c 16 -G a40
model="stel"
conda activate stel

# dataset="CrossNews_Article_Article.csv"
# dataset="CrossNews_Article_Tweet.csv"
# dataset="CrossNews_Tweet_Tweet.csv"
# dataset="CrossNews_Article_Article_short.csv"
dataset="CrossNews_Article_Tweet_short.csv"

params="default"

# dataset="empty.csv"
# params="no_train"

date
cd /nethome/mma81/storage/CrossNews
current_date=$(date +"%m_%d-%H_%M")
output_file="logs/${current_date}_${model}_${dataset}.txt"

python src/train_embedding.py \
--model ${model} \
--train \
--train_file verification_data/train/${dataset} \
--parameter_sets ${params} | tee $output_file
date
exit





salloc -c 16 -G a40
model="part"
conda activate part

# salloc -c 16 -G a40
# model="luar"
# conda activate luar

# salloc -c 16 -G a40
# model="stel"
# conda activate stel

# dataset="CrossNews_Article_Article.csv"
# dataset="CrossNews_Article_Tweet.csv"
dataset="CrossNews_Tweet_Tweet.csv"

date
cd /nethome/mma81/storage/CrossNews
current_date=$(date +"%m_%d-%H_%M")
output_file="logs/${current_date}_${model}_${dataset}.txt"
python src/run_embeddings.py ${model} ${dataset} | tee $output_file
date
exit



salloc -c 8

# dataset="CrossNews_Article.csv"
# dataset="CrossNews_Tweet.csv"
# dataset="CrossNews_Both.csv"
# dataset="CrossNews_Mini.csv"
# dataset="global_elite.csv"
dataset="global_standard.csv"

# model="luar_aa"
# model="part_aa"
# model="stel_aa"
model="ngram_aa"
# model="ppm_aa"

# conda activate luar
conda activate AuthorID_copy
cd /nethome/mma81/storage/CrossNews

date
python src/run_attribution.py \
--model ${model} \
--train \
--query_file attribution_data/query/${dataset} \
--parameter_sets default \
--test \
--target_file attribution_data/test/${dataset}
date

exit



salloc -c 8 -G a40

salloc -G a40:4

date
conda activate stel
cd /nethome/mma81/storage/CrossNews

# set="task_only_big"
set="prompt_av_big"
# set="lip_big"

python src/run_verification.py \
--model llm_prompting \
--train \
--train_file verification_data/train/empty.csv \
--parameter_sets ${set} \
--save_folder tweet_topic \
--test \
--test_files verification_data/test/global_elite_tweet.csv

exit

# --test_files verification_data/test/CrossNews_Tweet_Tweet.csv \
# verification_data/test/CrossNews_Article_Article.csv \
# verification_data/test/CrossNews_Article_Tweet.csv
date



salloc -G a40:8

# query="CrossNews_Article_llm.csv"
# query="CrossNews_Tweet_llm.csv"
# query="global_elite_llm.csv"
query="global_elite_llm_tweet.csv"

model="llm_prompting_aa"

# set="task_only"
# set="prompt_av"
# set="lip"
# set="task_only_big"
set="prompt_av_big"
# set="lip_big"

conda activate stel

date
python src/run_attribution.py \
--model ${model} \
--train \
--query_file attribution_data/query/${query} \
--parameter_sets ${set} \
--save_folder tweet_topic \
--test \
--target_file attribution_data/test/global_elite_llm_tweet.csv

date

exit




salloc -c 8

cd /nethome/mma81/storage/CrossNews

date

# conda activate AuthorID_copy
conda activate luar
# conda activate part
# conda activate stel



# dataset="CrossNews_Article_llm.csv"
dataset="CrossNews_Tweet_llm.csv"

model="luar_aa"
# model="part_aa"
# model="stel_aa"
# model="ngram_aa"
# model="ppm_aa"

date
python src/run_attribution.py \
--model ${model} \
--save_folder 20_author_results \
--train \
--query_file attribution_data/query/${dataset} \
--parameter_sets default \
--test \
--target_file attribution_data/test/CrossNews_llm.csv
date

exit


salloc -c 2

cd /nethome/mma81/storage/CrossNews

date
conda activate stel

# dataset="CrossNews_Article.csv"
# dataset="CrossNews_Tweet.csv"
# dataset="CrossNews_Both.csv"
dataset="global_standard.csv"
# dataset="global_elite.csv"
# dataset="global_standard_tweet.csv"
# dataset="global_elite_tweet.csv"

# set="mistral_test_1"
# set="mistral_prompt_av"
# set="mistral_prompt_lip"
# set="mistral_prompt_sem"
set="mistral_prompt_ignore"

model="llm_embedding_aa"
if [[ "$dataset" == "CrossNews_Article.csv" || "$dataset" == "CrossNews_Tweet.csv" || "$dataset" == "CrossNews_Both.csv" ]]; then
    test_dataset="CrossNews.csv"
else
    test_dataset="$dataset"
fi

date
python src/run_attribution.py \
--model ${model} \
--train \
--query_file attribution_data/query/${dataset} \
--parameter_sets ${set} \
--test \
--target_file attribution_data/test/${test_dataset}
date

exit
