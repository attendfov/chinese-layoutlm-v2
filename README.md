# Prepare
- 数据目录结构  
```shell script
data
|-- gartner_data
|   |-- data
|   |-- embedding_no_train
|   |-- models
|   `-- runs
`-- xfund-and-funsd
    |-- XFUND-and-FUNSD
    |-- models
    |-- ner_visualize
    |-- re_visualize
    `-- runs
```
- 数据集
目录gartner_data/data[链接](https://pan.baidu.com/s/1BFyGioxGDcR8Fw0S-1VyyQ?pwd=11m8)    
目录xfund-and-funsd/XFUND-and-FUNSD[链接](https://pan.baidu.com/s/1eoBvkkmM1bSSgkYyvVq6cQ?pwd=9pde)  
- 预训练模型layoutxlm  
目录pretrained_model/layoutxlm-base[链接](https://pan.baidu.com/s/1tFlF_-zzV45GL5eDlJVG5A?pwd=svbw)   

# Key-Value Pair抽取

- python run_xfun_ser.py 进行实体识别训练  
  `python run_xfun_ser.py 
  --model_name_or_path
    ./pretrained_model/layoutxlm-base
    --output_dir
    ./data/xfund-and-funsd/models/test-ner-xfund
    --do_train
    --do_eval
    --lang
    zh
    --num_train_epochs
    100
    --warmup_ratio
    0.1
    --fp16
    --additional_langs
    all
    --per_device_train_batch_size
    16
    --per_device_eval_batch_size
    16
    --logging_dir
    ./data/xfund-and-funsd/runs/ner-xfund
    --save_steps
    300
    --logging_steps
    300
    --evaluation_strategy
    steps
    --eval_steps
    300`
- python run_xfun_re.py 进行关系抽取训练，使用datasets.xfun_pipline.py
进行数据的读取，处理  
  `--model_name_or_path
./pretrained_model/layoutxlm-base
--output_dir
./data/xfund-and-funsd/models/test-re-xfund
--do_train
--do_eval
--lang
zh
--num_train_epochs
100
--warmup_ratio
0.1
--fp16
--additional_langs
all
--per_device_train_batch_size
16
--per_device_eval_batch_size
16
--logging_dir
./data/xfund-and-funsd/runs/re-xfund
--save_steps
300
--logging_steps
300
--evaluation_strategy
steps
--eval_steps
300
--learning_rate
3e-5`

- pred_data_process.py 在进行实体识别前，对bbox进行 行对齐
  - 使用行对齐操作，使得模型在推理阶段f1提升1.2%

- 实体预测
`python run_xfun_ser_predict.py  
--model_name_or_path
./data/xfund-and-funsd/models/test-ner-xfund/checkpoint-3900
--output_dir
./data/xfund-and-funsd/models/test-ner-xfund
--do_predict
--lang
zh
--warmup_ratio
0.1
--fp16`

- 基于实体识别结果的关系预测
`python run_xfun_re_predict.py 
--model_name_or_path
./data/xfund-and-funsd/models/test-re-xfund/checkpoint-10500
--output_dir
./data/xfund-and-funsd/models/test-re-xfund
--do_predict
--lang
zh
--warmup_ratio
0.1
--fp16`

- ner_visualize.py 识别出的实体和Ground True的可视化

- re_visualize.py 关系的可视化

- results_process_for_re.py  将对齐的实体识别结果处理成re输入格式，测试的不使用

- python app.py
  - 端到端KV关系抽取服务
  - 输入visual rich document，输出key value pair可视化结果
  
# Document Embedding
## 算法
- 基于SBERT中有监督学习训练算法

## Train mode
- 样例数据准备

- 模型训练
` python run_xfun_doc_embedding_train.py
--model_name_or_path
./pretrained_model/layoutxlm-base
--output_dir
./data/gartner_data/models
--do_train
--do_eval
--lang
zh
--num_train_epochs
100
--warmup_ratio
0.1
--fp16
--additional_langs
all
--per_device_train_batch_size
8
--per_device_eval_batch_size
8
--logging_dir
./data/gartner_data/runs
--save_steps
300
--logging_steps
300
--evaluation_strategy
steps
--eval_steps
300
--learning_rate
3e-5`


## No Train

- 文档向量化
` python run_xfun_doc_embedding_no_train
--model_name_or_path
./pretrained_model/layoutxlm-base
--output_dir
./data/gartner_data/embedding_no_train/
--do_predict
--lang
zh
--warmup_ratio
0.1
--fp16`