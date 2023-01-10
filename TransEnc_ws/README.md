# Transformer Encoder with Seen Course 

## train and predict

```bash
python run_transrec.py \
--device cuda:0 \
--num_workers 8 \
--dir [output dir] \
--do_save \
--predict_class course \
--train_file [train json file] \
--valid_file [valid json file] \
--test_file [test json file] \
--seen_filter [filter json file] \
--predict_file predict.csv \
--embed_dim 256 \
--batch_size 32 \
--hidden_dim 1024 \
--nhead 4 \
--num_layers 3 \
--dropout 0.25 \
--do_scheduler \
--warmup_step 10 \
--early_stop 10
```
