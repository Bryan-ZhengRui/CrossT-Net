# CrossT-Net：Cross Transformer for Efficient Lidar-Based Loop Closure Detection（Our paper is being submitted to RAL）
![image](https://user-images.githubusercontent.com/96043999/192136749-8d6608dd-4bc2-4689-bb2f-ecb775a2c2c2.png)


## How to use: 
We used the pytorch framework to build our network.
  
In order to train on a whole dataset, you need at least one gpu. 


* First, you'd better be able to install the main packages we need:
  
``` pip install -r requirements.txt ```

### How to train the dataset： 

Our proposed method uses three inputs (depth map, intensity map, normal map), which requires spherical projection processing of raw point clouds of the lidar. You could utilize [OverlapNet](https://github.com/PRBonn/OverlapNet) 's Demo1 to obtain these 2D images. 

The KITTI07 sequence we have here is the preprocessed data and can be directly used to input the network. Below is a visualization of a frame in the sequence07
 
![2d_maps](https://user-images.githubusercontent.com/96043999/192137282-5c8c88ea-5577-4aff-89be-1f950edc6ab7.png)
