> modify default.yaml for training
```
name: InsightEye
version: 1.0
task: "InducedClick"            #  change to your task name

label2name: "configs/label2name/inducedclick.json"    # must change to your label2name.json (!!!)

arch:
  type: efficentnet_b5     # choice one arch ['efficient_b5', 'efficient_b4', 'resnet_101' ... ]
  resume: ""               # if you want retrain from a checkpoint, please add the checkpoint path (!!!)
  best_model: ""           
  args:
    num_classes: 4         # consistent with your task （!!!）


loader:
  type: normdataloader
  args:
    img_fmt: rgb
    dataset_name: path_label
    train_data:
      data_file: "data/train.txt"         # change to your training dataset.txt
      is_train: true
      data_aug:                           # custom data argument
        image_resize: 380                 # consistent to your model
        random_crop: true
        random_rotate: false
        random_hflip: false
        random_vflip: false
        gaussianblur: false
        random_erase: false
        random_gamma: false
        random_brightness: false
        random_saturation: false
        random_adjust_hue: false
    valid_data:
      data_file: "data/valid.txt"         # change to your validation dataset.txt
      is_train: false
      data_aug:
        image_resize: 380                 # consistent to your model
    batch_size: 64                        # custom batch size
    pin_memory: true
    num_workers: 2                        # custom dataloader workers

solver:
  n_gpus: 8                               # set gpu numbers  (!!!)
  epochs: 120                             # custom your own training strategy (optimizer and lr_scheduler)
  optimizer:
    type: SGD
    args:
      lr: 0.01
      momentum: 0.9
  lr_scheduler:                           # custom your own lr_scheduler ['SGD', 'Adam'...]
    type: StepLR
    args:
      step_size: 50
      gamma: 0.1
  save_dir: "train_log"          # set checkpoint and log save dir
  ckpt_dir: "ckpt"               # checkpoint will save to $save_dir/$task_$arch.type/$start_time/ckpt
  log_dir: "log"                 # checkpoint will save to $save_dir/$task_$arch.type/$start_time/log
  save_freq:
  valid_interval: 1
  log_interval:
  verbosity: 1
  monitor: "valid_acc"            # set monitor: ['valid_acc', 'valid_auc', 'valid_loss'] 
  monitor_mode: "max"             # consistant to you monitor, ['max', 'min']
  tensorboardx: true              # use tensorboard or not, tensorboard dir: $log_dir/train/ or $logdir/infer/
  resume: ""
  fix_random_seed:

loss: class_balance_loss          # choice one loss: ['cross_entroy_loss', 'focal_loss', 'class_balance_loss']
metrics: common

```

> modify infer_default.yaml for inference
```
basic:
  name: InsightEye
  version: 1.0
  task: "InducedClick"                                 # change to your task name
  n_gpus: 1
  seed: 666
  id2name: "configs/label2name/inducedclick.json"      # must change to your label2name.json (!!!)

arch:
  type: efficentnet_b5
  resume:
  best_model: "..../ckpt/model_best_weight.pth"        # must change to your best model path (!!!)
  args:
    num_classes: 4                                     # consistent to your task

loader:
  data_aug:
    image_resize: 380                                  # consistent to your model
  type: normdataloader
  args:
    img_fmt: rgb
    dataset_name: path_label
    infer_data:
      data_file: "data/valid_imbalance.txt"
      is_train: false
      data_aug:
        image_resize: 380
    batch_size: 64
    pin_memory: true
    num_workers: 2
scheme:
  use_loader: false
  result_dir: ""
  tensorboardx: false

```