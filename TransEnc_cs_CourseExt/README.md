# Transformer Encoder with Cosine Similarity and with Course Extractor

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
--embed_dim 128 \
--nhead 2 \
--num_layers 2 \
--dropout 0.2 \
--do_scheduler \
--warmup_step 10 \
--batch_size 32 \
--early_stop 40
```
