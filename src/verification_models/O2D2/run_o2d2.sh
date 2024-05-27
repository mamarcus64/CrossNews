#!/bin/bash

# set variables
train_file=/nethome/mma81/storage/CrossNews/pairs/train_Article_Tweet.csv
train_genre_0=Article
train_genre_1=Tweet
adhominem_epochs=3
o2d2_epochs=12

# preprocessing
# cd preprocessing
# echo step0
# python step0_convert_to_jsonl.py $train_file $train_genre_0 $train_genre_1
# echo step1
# python step1_parse_and_split.py
# echo step2
# python step2_preprocess.py
# echo step3
# python step3_count.py
# echo step4
# python step4_make_vocabularies.py
# echo step5
# python step5_sample_pairs_cal.py
# echo step6
# python step6_sample_pairs_val.py
# echo step7
# python step7_sample_pairs_dev.py
# echo step8
# python step8_make_test_pairs.py
# cd ..

# train adhominem
cd training_adhominem
echo training adhominem
python train_adhominem.py -epochs $adhominem_epochs
cd ..

# train o2d2
cd training_o2d2
echo training o2d2
python train_o2d2.py -epoch_trained $((adhominem_epochs - 1)) -epochs $o2d2_epochs
cd ..

# inference
cd inference
echo running inference
python run_inference.py test_pairs_A_T $((o2d2_epochs - 1))
python run_inference.py test_pairs_T_T $((o2d2_epochs - 1))
python run_inference.py test_pairs_A_A $((o2d2_epochs - 1))

out_dir="$((adhominem_epochs - 1))_$((o2d2_epochs - 1))"
mkdir -p $out_dir
mv *.json $out_dir
mv *.log $out_dir