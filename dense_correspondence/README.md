## 1. Dependencies/Setup
* If you don't already have a virtualenvironment set up for this project, create a Python2 virtualenv with ```virtualev rope_env```
* ```source rope_env/bin/activate```
* ```cd path/to/dynamic-rope-sim/dense_correspondence```
* Install the dependencies needed (```pip install -r requirements.txt```)
* Change directories to ```/path/to/dynamic-rope-sim/dense_correspondence/pytorch-segmentation-detection/vision/torchvision/models``` and run ```cp resnet.py path/to/your_virtualenv/lib/python2.7/site-packages/torchvision/models/resnet.py``` (changing the path to your virtualenv as necessary)

* ^^NOTE: Without this step, when I ran ```python live_heatmap_visualization.py```, I got the following error: ```TypeError: __init__() got an unexpected keyword argument 'fully_conv'```
* ^^TODO: there should be a way to avoid this using ```sys.path.insert``` in the script, though I haven't configured it yet

## 2. Steps for evaluating a trained network
### Copying over the trained network:
* To run the network, we need the ```.pth``` files (only the checkpoint for the iteration you want, typically 3500, not all!) and the ```.yaml``` files. In the folder ```networks``` make a new folder for the name of the netork you want. ```scp``` or ```rsync``` over the relevant ```.pth``` and ```.yaml``` files from your remote machine into this folder. (See script ```copy_network.py``` which I use. You can probably use it directly but substitute my username for yours and change the path to the networks accordingly.)
### Running the heatmap visualization:
* ```cd tools```
* Open ```live_heatmap_visualization.py``` and at the bottom, set ```network_dir``` and ```image_dir``` to the relevant paths
* Run ```python live_heatmap_visualization.py```

## 3. Steps for getting dependencies set up in Blender's python.
* Blender comes bundled with its own separate installation of Python entirely, which is different from your system Python ```/usr/bin/python```
* ```cd``` to the directory in which you downloaded Blender (for me, that is: ```/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app```)
* ```cd Contents/Resources/2.80/python/bin```
* Now, you should see a bunch of ```pip```'s installed here. We are going to use ```pip```, not ```pip3``` or ```pip3.7```
* We need to install the dependencies for this project using the ```pip``` here, not your usual ```pip``` on your system. Luckily this is very easy!
* Run ```./pip install -r path/to/dynamic-rope-sim/dense_correspondence/requirements.txt``` (Note the ```./```, this ensures that we use Blender's ```pip``` instead of your ```pip```)
* Finally, that ugly torch fix. Ugh. Run ```cp path/to/dynamic-rope-sim/dense_correspondence/pytorch-segmentation-detection/vision/torchvision/models/resnet.py path/to/blender.app/Contents/Resources/2.80/python/lib/python3.7/site-packages/torchvision/models/``` to install the proper resnet.py from our custom version of torch, to Blender version of torch. This is super ugly, we should find a fix.
* If you get a torch error along the lines of `Library not loaded: @rpath/libc++.1.dylib`, try running `install_name_tool -add_rpath /usr/lib path/to/Blender.app/Contents/Resources/2.80/python/lib/python3.7/site-packages/torch/_C.cpython-37m-darwin.so` taken from https://discuss.pytorch.org/t/installation-problem-library-not-loaded-rpath-libc-1-dylib/36802/5
