# Multi-layer Perceptron with Feature Mask

## train and predict

```bash
python onehot.py \
--device cuda:0 \
--num_workers 8 \
--dir [output dir] \
--do_save \
--predict_class {course, subgroup} \
--train_file [train json file] \ 
--valid_file [valid json file] \
--test_file [test json file] \
--seen_filter [filter json file] \
--predict_file predict.csv
```
