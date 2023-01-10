
```bash
python run_nfm.py \
--device cuda:0 \
--num_workers 8 \
--dir [output dir] \
--do_save \
--predict_class {course, subgroup} \
--train_file [train json file] \
--valid_file [valid json file] \
--test_file [test json file] \
--predict_file predict.csv \
--embed_dim 128 \
--dropout 0.2
```
