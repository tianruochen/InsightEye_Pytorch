- Prepare your own training data and valid data.
- a train.txt and valid.txt is necessary,format:
```
/data1/changqing/InduceClick_Data/images/0000001.jpg	0
/data1/changqing/InduceClick_Data/images/0000002.jpg	3
/data1/changqing/InduceClick_Data/images/0000003.jpg	0
/data1/changqing/InduceClick_Data/images/0000004.jpg	0
```
- a json file that mapping label to class_name is necessary, and save it to 'configs/label2name/', like this:

``` 
# configs/label2name/inducedclick.json
{
    "0": "induce_button",
    "1": "induce_videos",
    "2": "induce_honbao",
    "3": "normal_images"
}
```
- some useful scripts:
```
build_data.py  -- build train.txt and valid.txt form root images dir
check_image.py  -- check image and filtered invalid images 
...
```

