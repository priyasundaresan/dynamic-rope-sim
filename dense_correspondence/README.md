## 1. Dependencies/Setup
* Install or ensure you have the dependencies needed (```pip install -r requirements.txt```)
* Change directories to ```/path/to/dynamic-rope/dense_correspondence/pytorch-segmentation-detection/vision/torchvision/models``` and run ```cp resnet.py path/to/your_virtualenv/lib/python2.7/site-packages/torchvision/models/resnet.py``` (changing the path to your virtualenv as necessary)

* ^^NOTE: Without this step, when I ran ```python live_heatmap_visualization.py```, I got the following error: ```TypeError: __init__() got an unexpected keyword argument 'fully_conv'```
* ^^TODO: there should be a way to avoid this using ```sys.path.insert``` in the script, though I haven't configured it yet

## 2. Steps for evaluating a trained network
### Copying over the trained network:
* To run the network, we need the ```.pth``` files (only the checkpoint for the iteration you want, typically 3500, not all!) and the ```.yaml``` files. In the folder ```networks``` make a new folder for the name of the netork you want. ```scp``` or ```rsync``` over the relevant ```.pth``` and ```.yaml``` files from your remote machine into this folder. (See script ```copy_network.py``` which I use. You can probably use it directly but substitute my username for yours and change the path to the networks accordingly.)
### Running the heatmap visualization:
* ```cd tools```
* Open ```live_heatmap_visualization.py``` and at the bottom, set ```network_dir``` and ```image_dir``` to the relevant paths
* Run ```python live_heatmap_visualization.py```
