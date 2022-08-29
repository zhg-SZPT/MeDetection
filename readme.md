# Adaptive Vision Detection of Industrial Product Defects
The code for the paper "Adaptive Vision Detection of Industrial Product Defects".
## Requirements
python 3.8 <br>
pytorch 1.10.10 <br>
cuda 11.2 <br>
l2l 0.1.7
## Dataset Preparation
1. Dataset Download Address <https://pan.baidu.com/s/1GgQhuj6it2e4UtpuOtVzlA?pwd=4a92 
提取码：4a92> Put the downloaded dataset into the dataset folder <br>
2. Our dataset contains 30 different categories of industrial products, sourced from MVTec, DAGM and our own cap dataset. 
If you need our dataset, please contact our corresponding author and you will be required to sign a confidentiality agreement.
Corresponding author email : zhg2018@sina.com.
  ```python
  
  地毯
  人行道
  瓶盖
  药片
  ...
  ```

## MAMl Train
1. Run maml.python file for maml training First remove the target task from the dataset by the following code, then set the maml related parameters
```python

    tasknames1=["电缆"]
...
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--fast_lr', type=float, help='task-level inner update learning rate', default=1e-5)
    argparser.add_argument('--meta_bsz', type=int, help='task_batch', default=1)
    argparser.add_argument('--adaptation_steps', type=int, help='inner loop iter', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=29)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
```
## Train
1.After maml training, add in the weights for basic training to run basetrain.py, data loading and parameter settings are as follows:
  ```python
     argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=2000)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-5)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
  ```
## Evaluate
Load the training results for evaluation and display the results as a pdf file
