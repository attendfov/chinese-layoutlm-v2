

# Prepare
- 数据目录结构  
```shell script
当前工作目录/DATA/
├── gartner_data
│   └── data 
├── pretrained-models
│   └── layoutxlm-base
├── xfund-and-funsd
│   └── XFUND-and-FUNSD

```
- 数据集  
  - 个人数据，目录 gartner_data/data[链接](https://pan.baidu.com/s/1BFyGioxGDcR8Fw0S-1VyyQ?pwd=11m8)    
  - 开源xfund和funsd数据，合并成了一个格式统一的数据集，目录 xfund-and-funsd/XFUND-and-FUNSD[链接](https://pan.baidu.com/s/1eoBvkkmM1bSSgkYyvVq6cQ?pwd=9pde)  
  
- 预训练模型layoutxlm  
目录 pretrained-models/layoutxlm-base[链接](https://pan.baidu.com/s/1tFlF_-zzV45GL5eDlJVG5A?pwd=svbw)   

# Key-Value Pair抽取
## steps
1. python run_xfun_ser.py 进行实体识别训练  
`python run_xfun_ser.py --model_name_or_path  ../DATA/pretrained-models/layoutxlm-base --output_dir ../DATA/xfund-and-funsd/models/test-ner-xfund --logging_dir ../DATA/xfund-and-funsd/runs/ner-xfund --do_train --do_eval --lang zh --num_train_epochs 100  --warmup_ratio 0.1 --additional_langs all --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --save_steps 300 --logging_steps 300 --evaluation_strategy steps --eval_steps 300`  

实体识别训练时如果使用fp16，训练一段时间后loss会出现NaN

2. python run_xfun_re.py 进行关系抽取训练，使用datasets.xfun_pipline.py
进行数据的读取，处理   
`python run_xfun_re.py --model_name_or_path ../DATA/pretrained-models/layoutxlm-base --output_dir ../DATA/xfund-and-funsd/models/test-re-xfund --logging_dir ../DATA/xfund-and-funsd/runs/re-xfund --do_train --do_eval --lang zh --num_train_epochs 100 --warmup_ratio 0.1 --additional_langs all --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --save_steps 300 --logging_steps 300 --evaluation_strategy steps --eval_steps 300 --learning_rate 3e-5 --fp16`

3. 可选(Optional) 执行pred_data_process.py，会在进行实体识别前，对bbox进行 行对齐  
  使用行对齐操作，使得模型在推理阶段f1提升1.2%  
  改操作，会生成文件：当前工作目录/DATA/xfund-and-funsd/XFUND-and-FUNSD/zh.val.align.json，
  同时在预测时，修改`工作目录/layoutlmft/data/datasets/xfun_predict.py`加载改文件

4. 实体预测  
`python run_xfun_ser_predict.py  --model_name_or_path ../DATA/xfund-and-funsd/models/test-ner-xfund/checkpoint-6300 --output_dir ../DATA/xfund-and-funsd/models/test-ner-xfund --do_predict --lang zh`

5. 基于实体识别结果的关系预测  
`python run_xfun_ser_predict.py  --model_name_or_path ../DATA/xfund-and-funsd/models/test-ner-xfund/checkpoint-6300 --output_dir ../DATA/xfund-and-funsd/models/test-ner-xfund --do_predict --lang zh`

6. 将识别出的实体和Ground Truth进行可视化  
`python ner_visualize.py`  

7. key-value关系识别的可视化  
`python re_visualize.py `

## file explain
1. results_process_for_re.py  将对齐的实体识别结果处理成re输入格式，测试的不使用  
2. python app.py  
a. 端到端KV关系抽取服务      
b. 输入visual rich document，输出key value pair可视化结果  

# Document Embedding
## 算法
- 基于Sentence-BERT中有监督学习训练算法

## Train mode  
- 样例数据准备  
- 模型训练  
`python run_xfun_doc_embedding_train.py --model_name_or_path ../DATA/pretrained-models/layoutxlm-base --output_dir ../DATA/gartner-data/models --logging_dir ../DATA/gartner-data/runs --do_train --do_eval --lang zh --num_train_epochs 100 --warmup_ratio 0.1 --fp16 --additional_langs all --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --save_steps 300 --logging_steps 300 --evaluation_strategy steps --eval_steps 300 --learning_rate 3e-5`

## No Train mode
- 文档向量化
` python run_xfun_doc_embedding_no_train.py --model_name_or_path ../DATA/pretrained-models/layoutxlm-base --output_dir ../DATA/gartner-data/embedding_no_train/ --do_predict --lang zh --warmup_ratio 0.1 --fp16`

## Tips
1. 修改数据的预处理方式之后，需要清除缓存~/.cache/huggingface/datasets，否则模型会直接使用缓存中的数据，造成数据修改失败


Part of the code from [link](https://github.com/microsoft/unilm/tree/master/layoutlmft)