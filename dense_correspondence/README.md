1. Dependencies/Setup
- I froze the dependencies I needed to run all the code in this repo into a requirements.txt, so if you're starting from scratch you can make a virtualenv and pip install directly from that, or if you already have a virtualenv, you can use my requirements.txt to get the version specifics. 

- At this point, you have everything needed to work with the pytorch code, but for some reason, the Dense Object Nets authors insist on using this library pytorch-segmentation-detection which is meant to override some of the normal torch source code. 

- Change directories one level higher into pytorch-segmentation-detection. Go to pytorch_segmentation_detection/vision/torchvision/models and run 'cp resnet.py path/to/your_virtualenv/lib/python2.7/site-packages/torchvision/models/resnet.py' (changing the path to your virtualenv as necessary)

- The reason the above step was necessary for me is that without it, when I ran 'python live_heatmap_visualization.py', I got the following error: "TypeError: __init__() got an unexpected keyword argument 'fully_conv'" 

^^TODO: there should be a way to avoid this using sys.path.insert in the script, though I haven't configured it yet

2. Steps for evaluating a trained network

Copying over the trained network:
- To run the network, we need the .pth files (only the checkpoint for the iteration you want, not all!) and the .yaml files. In the folder 'networks' make a new folder for the name of the netork you want. Copy over the relevant .pth and .yaml files from your remote machine into this folder. (See script copy_network.py for an example)

Running the heatmap visualization:
- Change directories into 'tools'
- Open live_heatmap_visualization.py and at the bottom, set network_dir and image_dir to the relevant paths
- Run python live_heatmap_visualization.py
