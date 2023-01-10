#! /bin/bash
export TOKENIZERS_PARALLELISM=False
python train.py \
--device cuda:1 \
--num_workers 4 \
--dir multitask_unseen_course \
--do_save \
--do_log \
--predict_class course \
--course_file ../../hahow/courses.json \
--course_map ../course_map.json \
--train_file ../../hahow/train_users.json \
--valid_file ../../hahow/val_seen_users.json \
--test_file ../../hahow/test_seen_users.json \
--predict_file predict.csv \
--seen_filter ../../hahow/train_users.json ../../hahow/val_seen_users.json \
--model_path ckiplab/bert-tiny-chinese \
--epoch 3 \
--steps 100 \
--max_length 256 \
--warmup_step 100
