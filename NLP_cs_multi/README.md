# NLP with Cosine Similarity with Multitask

## train and predict

# 
```bash
export TOKENIZERS_PARALLELISM=False
python train.py \
--device cuda:0 \
--num_workers 4 \
--dir [output dir] \
--do_save \
--do_log \
--predict_class {course, subgroup} \
--course_file [course json file] \
--course_map course_map.json \
--train_file [train json file] \
--valid_file [valid json file] \
--test_file [test json file] \
--seen_filter [filter json file] \
--predict_file predict.csv \
--model_path ckiplab/bert-tiny-chinese \
--epoch 3 \
--steps 100 \
--max_length 256 \
--warmup_step 100
```
