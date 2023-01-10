#! /bin/bash
python run_nfm.py \
--device cuda:2 \
--num_workers 8 \
--dir nfm_unseen_group \
--do_save \
--predict_class subgroup \
--train_file ../../hahow/train_users.json \
--valid_file ../../hahow/val_unseen_users.json \
--test_file ../../hahow/test_unseen_users.json \
--predict_file predict.csv \
--embed_dim 256 \
--early_stop 20
