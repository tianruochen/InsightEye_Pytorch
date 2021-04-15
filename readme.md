# InsightEye_Pytorch
[![python version](https://img.shields.io/badge/python-3.6%2B-brightgreen)]()
[![coverage](https://img.shields.io/badge/coverage-56%25-orange)]()

Focus on image classification.

## Table of Contents

- [Structure](#structure)
- [Data](#Data)
- [Train](#Train)
- [Inference](#Inference)
- [Web Service](#Web Service)
- [Contributing](#contributing)
- [License](#license)



## structure
```
├── config                     
│     ├── database_config          some config files for database (web service)
│     ├── label2name               label to class name (json file)
│     ├── logger_config            some config files for logging (not use)
│     ├── model_config             some config files for model training and inference
├── data                           
│     ├── images                   a directory of original images for trianing and validation
│     ├── train.txt                a file that stores training images and corresponding labels
│     ├── valid.txt                a file that stores validation images and corresponding labels
├── modules 
│     ├── __init__.py                    
│     ├── datasets 
│     │       ├── argument.py                  data argument
│     │       ├── dataset.py                   dataset
│     │       ├── dataloader.py                data argument 
│     ├── losses 
│     │       ├── cs_loss.py                   cross_entropy_loss
│     │       ├── focal_loss.py                focal loss
│     │       ├── class_balance_loss.py        class balance loss
│     ├── models 
│     │       ├── base_model.py                 
│     │       ├── resnet.py                    to do
│     │       ├── vggnet.py                    to do 
│     │       ├── xception.py                  to do
│     │       ├── mobilenetv2.py               to do
│     │       ├── efficientnet.py              supported  
│     ├── trainer 
│     │       ├── base.py                      
│     │       ├── trainer.py                   deal with the whole training process
│     │       ├── inferer.py                   deal with the whole inference process
│     ├── utils 
│     │       ├── comm_util.py                 common utils
│     │       ├── config_util.py               used to parse configuration files
│     │       ├── data_trans.py                data transform utils
│     │       ├── flops_counter.py             flops counter    
│     │       ├── server_util.py               server utils
│     │       ├── summary_utils.py             model summary utils=  
│     │       ├── visualization.py             tensorboardx visualization utils  
│     ├── train_log 
│     │       ├── ckpt                         a directory to save checkpoint
│     │       ├── log                          a directory to save log    
├── train_net.py                 training script      
├── infer_net.py                 inference script   
├── flask_server.py              web service script   
├── server_test.py               web service test script  
```     
## Data
a train.txt and a valid.txt is necessary.

format(path   label)(Separated by '\t'): 
```
/data1/changqing/InduceClick_Data/images/0000001.jpg	0
/data1/changqing/InduceClick_Data/images/0000002.jpg	3
/data1/changqing/InduceClick_Data/images/0000003.jpg	0
/data1/changqing/InduceClick_Data/images/0000004.jpg	0
```
## Train
To train a new model, use the main.py script.

> use default params (all parameters setted in model_config file)
```use default params (all parameters setted in model_config file)
python train_net.py --model_config XXX 
```
> use custom params
```use custom params
python train_net.py --model_config XXX --batch_size XXX --learning_rate XXX ...
# custom params
# --batch_size: int, [training batch size, None to use config setting]
# --learning_rate: float, [training learning rate, None to use config setting]
# --resume: str, [path to pretrain weights]
# --n_gpu: int, [the number of gpus]
# --epoch: int, [epoch number, 0 for read from config file]
# --save_dir: str, [directory name to save train snapshoot, None to use config setting]
# --valid_interval: str, [validation epoch interval, 0 for no validation]
# --log_interval: str, [mini batch interval to log]
# --fix_random_seed: bool, [If set True, set rand seed]
```

## Inference
Use the following command to inference.
> predict one image
```
python infer_net.py --infer_config XXX --input XXX.jpg
```
> predict few images
```
python infer_net.py --infer_config XXX --input XXX.jpg XXX.jpg XXX.jpg ...
```
> predict many images
```
python infer_net.py --infer_config XXX --input XXX.txt
# xxx.txt:
image1.jpg
image2.jpg
...
```
> other params
```
python infer_net.py --infer_config XXX --input XXX.txt --n_gpus XXX --best_model XXX
# --n_gpus: int, [the numbers of gpu needed (default 1)]
# --best_model: str, [the best model for inference]
```

## Web Service
> usage:  
```
python flask_server.py
```
refer: [here](https://doc2.ixiaochuan.cn/pages/viewpage.action?pageId=8410829)

## how to train on your own dataset
 - build your dataset (please refer [build_data.md](data/build_data.md))
 - modify config file (please refer [modify_config.md](configs/model_config/modify_config.md))
 - run 'python train_net.py'
## Contributing

## License

[MIT © Richard McRichface.](../LICENSE)