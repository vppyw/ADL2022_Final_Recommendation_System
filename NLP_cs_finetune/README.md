# NLP with Cosine Similarity Finetune

## train and predict

```bash
export TOKENIZERS_PARALLELISM=False
python train.py \
--device cuda:0 \
--num_workers 4 \
--dir [output dir] \
--do_save \
--do_log \
--model_path ckiplab/bert-tiny-chinese \
--predict_class {course, subgroup} \
--course_file [course json file] \
--course_map course_map.json \
--train_file [train json file] \
--valid_file [valid json file] \
--test_file [test json file] \
--predict_file predict.csv \
--seen_filter [filter json file] \
--epoch 3 \
--steps 500 \
--max_length 256 \
--batch_size 16 \
--course_batch_size 16 \
--gradient_accumulate 8 \
--lr 8e-5 \
--warmup 5000
```
