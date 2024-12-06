
salloc -c 8 -G a40
model="stel_av"
conda activate stel

# dataset="CrossNews_Article_Article.csv"
# dataset="CrossNews_Article_Tweet.csv"
dataset="CrossNews_Tweet_Tweet.csv"


# params="exp1_A_A"
# params="exp1_A_T"
# params="exp1_T_T"
# params="exp2_A_A"
# params="exp2_A_T"
params="exp2_T_T"

# save_folder="runs/experiment_1"
save_folder="runs/experiment_2"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Article_Tweet.csv"

params="${dataset}"

save_folder="runs/experiment_1"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Tweet_Tweet.csv"

params="${dataset}"

save_folder="runs/experiment_1"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Article_Article.csv"

params="${dataset}"

save_folder="runs/experiment_2"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Article_Tweet.csv"

params="${dataset}"

save_folder="runs/experiment_2"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Tweet_Tweet.csv"

params="${dataset}"

save_folder="runs/experiment_2"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Article_Article.csv"

params="${dataset}"

save_folder="runs/experiment_3"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Article_Tweet.csv"

params="${dataset}"

save_folder="runs/experiment_3"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Tweet_Tweet.csv"

params="${dataset}"

save_folder="runs/experiment_3"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Article_Article.csv"

params="${dataset}"

save_folder="runs/experiment_4"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Article_Tweet.csv"

params="${dataset}"

save_folder="runs/experiment_4"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel



dataset="CrossNews_Tweet_Tweet.csv"

params="${dataset}"

save_folder="runs/experiment_4"


    
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
| tee $output_file
date
exit

###########################


    
salloc -c 8 -G a40
model="stel_av"
conda activate stel


