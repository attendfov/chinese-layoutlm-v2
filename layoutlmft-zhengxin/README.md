# Run

- 实体识别  
`python run_xfun_ser.py --model_name_or_path
../../data/pretrained-models/layoutxlm-base
--output_dir
../../data/zhengxin/data/标注批1部分批2/models/ner
--do_train
--do_eval
--lang
zh
--num_train_epochs
100
--warmup_ratio
0.1
--fp16
--per_device_train_batch_size
16
--per_device_eval_batch_size
16
--logging_dir
../../data/zhengxin/data/标注批1部分批2/runs/ner
--save_steps
30
--logging_steps
30
--evaluation_strategy
steps
--eval_steps
30`