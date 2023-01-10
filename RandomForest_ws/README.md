# Random Forest with Feature Mask

## train and predict
```bash
python decision_tree.py \
--gpu [gpu device number] \
--dir [output dir] \
--do_save \
--predict_class {course, subgroup} \
--mask_num [mask number] \
--mask_p [mask ratio] \
--train_file [train json file] \
--valid_file [valid json file] \
--test_file [test json file] \
--seen_filter [filter json file] \
--predict_file predict.csv
```
