# CrossT-Net：Cross Transformer for Lidar-Based Loop Closure Detection
![image](https://user-images.githubusercontent.com/96043999/192136749-8d6608dd-4bc2-4689-bb2f-ecb775a2c2c2.png)


## How to use: 
We used the pytorch framework to build our network.
  
In order to train on a whole dataset, you need at least one gpu. 


* First, you'd better be able to install the main packages we need:
  
```
pip install -r requirements.txt 
```

### 1.How to train the dataset： 

Our proposed method uses three inputs (depth map, intensity map, normal map), which requires spherical projection processing of raw point clouds of the lidar. You could utilize **[OverlapNet](https://github.com/PRBonn/OverlapNet) 's** Demo1 to obtain these 2D images. 

The KITTI07 sequence we have here is the preprocessed data and can be directly used to input the network. Below is a visualization of a frame in the sequence07
 
![无标题](https://user-images.githubusercontent.com/96043999/192137765-17fa58c6-391b-4139-9c41-f85ec5991975.png)

The attention module in the module borrows from **[TransT](https://github.com/chenxin-dlut/TransT)** and changed in some regions.

If you want to train the dataset on a **single GPU**, you just need to write this command in the root directory:

``` 
python main_single_gpu.py
```

Or If you want to train the dataset on **multiple GPUs**, you first have to configure the GPU index and number in the [config file](https://github.com/Bryan-ZhengRui/CrossT-Net/blob/main/config/configfile.yaml), and then:

```
python main_multi_gpu.py 
```

### 2.How to test the dataset： 

Configure the parameters in the [config file](https://github.com/Bryan-ZhengRui/CrossT-Net/blob/main/config/configfile.yaml), and then:

``` 
python test.py
```

And after test, you will get an error distribution plot and predict.txt, which can be used for later P-R curve drawing.

# License

CrossT-Net is released under MIT License.

