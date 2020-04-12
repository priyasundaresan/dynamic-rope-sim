torch-vision
============

.. image:: https://travis-ci.org/pytorch/vision.svg?branch=master
    :target: https://travis-ci.org/pytorch/vision

This repository consists of:

-  `vision.datasets <#datasets>`__ : Data loaders for popular vision
   datasets
-  `vision.models <#models>`__ : Definitions for popular model
   architectures, such as AlexNet, VGG, and ResNet and pre-trained
   models.
-  `vision.transforms <#transforms>`__ : Common image transformations
   such as random crop, rotations etc.
-  `vision.utils <#utils>`__ : Useful stuff such as saving tensor (3 x H
   x W) as image to disk, given a mini-batch creating a grid of images,
   etc.

Installation
============

Anaconda:

.. code:: bash

    conda install torchvision -c soumith

pip:

.. code:: bash

    pip install torchvision

From source:

.. code:: bash

    python setup.py install

Datasets
========

The following dataset loaders are available:

-  `MNIST <#mnist>`__
-  `COCO (Captioning and Detection) <#coco>`__
-  `LSUN Classification <#lsun>`__
-  `ImageFolder <#imagefolder>`__
-  `Imagenet-12 <#imagenet-12>`__
-  `CIFAR10 and CIFAR100 <#cifar>`__
-  `STL10 <#stl10>`__
-  `SVHN <#svhn>`__
-  `PhotoTour <#phototour>`__

Datasets have the API: - ``__getitem__`` - ``__len__`` They all subclass
from ``torch.utils.data.Dataset`` Hence, they can all be multi-threaded
(python multiprocessing) using standard torch.utils.data.DataLoader.

For example:

``torch.utils.data.DataLoader(coco_cap, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)``

In the constructor, each dataset has a slightly different API as needed,
but they all take the keyword args:

-  ``transform`` - a function that takes in an image and returns a
   transformed version
-  common stuff like ``ToTensor``, ``RandomCrop``, etc. These can be
   composed together with ``transforms.Compose`` (see transforms section
   below)
-  ``target_transform`` - a function that takes in the target and
   transforms it. For example, take in the caption string and return a
   tensor of word indices.

MNIST
~~~~~
``dset.MNIST(root, train=True, transform=None, target_transform=None, download=False)``

``root``: root directory of dataset where ``processed/training.pt`` and ``processed/test.pt`` exist

``train``: ``True`` - use training set, ``False`` - use test set.

``transform``: transform to apply to input images

``target_transform``: transform to apply to targets (class labels)

``download``: whether to download the MNIST data


COCO
~~~~

This requires the `COCO API to be
installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`__

Captions:
^^^^^^^^^

``dset.CocoCaptions(root="dir where images are", annFile="json annotation file", [transform, target_transform])``

Example:

.. code:: python

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    cap = dset.CocoCaptions(root = 'dir where images are',
                            annFile = 'json annotation file',
                            transform=transforms.ToTensor())

    print('Number of samples: ', len(cap))
    img, target = cap[3] # load 4th sample

    print("Image Size: ", img.size())
    print(target)

Output:

::

    Number of samples: 82783
    Image Size: (3L, 427L, 640L)
    [u'A plane emitting smoke stream flying over a mountain.',
    u'A plane darts across a bright blue sky behind a mountain covered in snow',
    u'A plane leaves a contrail above the snowy mountain top.',
    u'A mountain that has a plane flying overheard in the distance.',
    u'A mountain view with a plume of smoke in the background']

Detection:
^^^^^^^^^^

``dset.CocoDetection(root="dir where images are", annFile="json annotation file", [transform, target_transform])``

LSUN
~~~~

``dset.LSUN(db_path, classes='train', [transform, target_transform])``

-  ``db_path`` = root directory for the database files
-  ``classes`` =
-  ``'train'`` - all categories, training set
-  ``'val'`` - all categories, validation set
-  ``'test'`` - all categories, test set
-  [``'bedroom_train'``, ``'church_train'``, ...] : a list of categories to
   load

CIFAR
~~~~~

``dset.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)``

``dset.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)``

-  ``root`` : root directory of dataset where there is folder
   ``cifar-10-batches-py``
-  ``train`` : ``True`` = Training set, ``False`` = Test set
-  ``download`` : ``True`` = downloads the dataset from the internet and
   puts it in root directory. If dataset is already downloaded, does not do
   anything.

STL10
~~~~~

``dset.STL10(root, split='train', transform=None, target_transform=None, download=False)``

-  ``root`` : root directory of dataset where there is folder ``stl10_binary``
-  ``split`` : ``'train'`` = Training set, ``'test'`` = Test set, ``'unlabeled'`` = Unlabeled set,
    ``'train+unlabeled'`` = Training + Unlabeled set (missing label marked as ``-1``)
-  ``download`` : ``True`` = downloads the dataset from the internet and
    puts it in root directory. If dataset is already downloaded, does not do
    anything.

SVHN
~~~~

``dset.SVHN(root, split='train', transform=None, target_transform=None, download=False)``

-  ``root`` : root directory of dataset where there is folder ``SVHN``
-  ``split`` : ``'train'`` = Training set, ``'test'`` = Test set, ``'extra'`` = Extra training set
-  ``download`` : ``True`` = downloads the dataset from the internet and
    puts it in root directory. If dataset is already downloaded, does not do
    anything.

ImageFolder
~~~~~~~~~~~

A generic data loader where the images are arranged in this way:

::

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/asd932_.png

``dset.ImageFolder(root="root folder path", [transform, target_transform])``

It has the members:

-  ``self.classes`` - The class names as a list
-  ``self.class_to_idx`` - Corresponding class indices
-  ``self.imgs`` - The list of (image path, class-index) tuples

Imagenet-12
~~~~~~~~~~~

This is simply implemented with an ImageFolder dataset.

The data is preprocessed `as described
here <https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset>`__

`Here is an
example <https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py#L48-L62>`__.

PhotoTour
~~~~~~~~~

**Learning Local Image Descriptors Data**
http://phototour.cs.washington.edu/patches/default.htm

.. code:: python

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    dataset = dset.PhotoTour(root = 'dir where images are',
                             name = 'name of the dataset to load',
                             transform=transforms.ToTensor())

    print('Loaded PhotoTour: {} with {} images.'
          .format(dataset.name, len(dataset.data)))

Models
======

The models subpackage contains definitions for the following model
architectures:

-  `AlexNet <https://arxiv.org/abs/1404.5997>`__: AlexNet variant from
   the "One weird trick" paper.
-  `VGG <https://arxiv.org/abs/1409.1556>`__: VGG-11, VGG-13, VGG-16,
   VGG-19 (with and without batch normalization)
-  `ResNet <https://arxiv.org/abs/1512.03385>`__: ResNet-18, ResNet-34,
   ResNet-50, ResNet-101, ResNet-152
-  `SqueezeNet <https://arxiv.org/abs/1602.07360>`__: SqueezeNet 1.0, and
   SqueezeNet 1.1

You can construct a model with random weights by calling its
constructor:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    squeezenet = models.squeezenet1_0()

We provide pre-trained models for the ResNet variants, SqueezeNet 1.0 and 1.1,
and AlexNet, using the PyTorch `model zoo <http://pytorch.org/docs/model_zoo.html>`__.
These can be constructed by passing ``pretrained=True``:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)


All pre-trained models expect input images normalized in the same way, i.e.
mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected
to be atleast 224.

The images have to be loaded in to a range of [0, 1] and then
normalized using `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`

An example of such normalization can be found in `the imagenet example here` <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>

Transforms
==========

Transforms are common image transforms. They can be chained together
using ``transforms.Compose``

``transforms.Compose``
~~~~~~~~~~~~~~~~~~~~~~

One can compose several transforms together. For example.

.. code:: python

    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                              std = [ 0.229, 0.224, 0.225 ]),
    ])

Transforms on PIL.Image
~~~~~~~~~~~~~~~~~~~~~~~

``Scale(size, interpolation=Image.BILINEAR)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rescales the input PIL.Image to the given 'size'. 

If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.

If 'size' is a number, it will indicate the size of the smaller edge. 
For example, if height > width, then image will be rescaled to (size \*
height / width, size) - size: size of the smaller edge - interpolation:
Default: PIL.Image.BILINEAR

``CenterCrop(size)`` - center-crops the image to the given size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Crops the given PIL.Image at the center to have a region of the given
size. size can be a tuple (target\_height, target\_width) or an integer,
in which case the target will be of a square shape (size, size)

``RandomCrop(size, padding=0)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Crops the given PIL.Image at a random location to have a region of the
given size. size can be a tuple (target\_height, target\_width) or an
integer, in which case the target will be of a square shape (size, size)
If ``padding`` is non-zero, then the image is first zero-padded on each
side with ``padding`` pixels.

``RandomHorizontalFlip()``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Randomly horizontally flips the given PIL.Image with a probability of
0.5

``RandomSizedCrop(size, interpolation=Image.BILINEAR)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the
original size and and a random aspect ratio of 3/4 to 4/3 of the
original aspect ratio

This is popularly used to train the Inception networks - size: size of
the smaller edge - interpolation: Default: PIL.Image.BILINEAR

``Pad(padding, fill=0)``
^^^^^^^^^^^^^^^^^^^^^^^^

Pads the given image on each side with ``padding`` number of pixels, and
the padding pixels are filled with pixel value ``fill``. If a ``5x5``
image is padded with ``padding=1`` then it becomes ``7x7``

Transforms on torch.\*Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Normalize(mean, std)``
^^^^^^^^^^^^^^^^^^^^^^^^

Given mean: (R, G, B) and std: (R, G, B), will normalize each channel of
the torch.\*Tensor, i.e. channel = (channel - mean) / std

Conversion Transforms
~~~~~~~~~~~~~~~~~~~~~

-  ``ToTensor()`` - Converts a PIL.Image (RGB) or numpy.ndarray (H x W x
   C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W)
   in the range [0.0, 1.0]
-  ``ToPILImage()`` - Converts a torch.\*Tensor of range [0, 1] and
   shape C x H x W or numpy ndarray of dtype=uint8, range[0, 255] and
   shape H x W x C to a PIL.Image of range [0, 255]

Generic Transforms
~~~~~~~~~~~~~~~~~~

``Lambda(lambda)``
^^^^^^^^^^^^^^^^^^

Given a Python lambda, applies it to the input ``img`` and returns it.
For example:

.. code:: python

    transforms.Lambda(lambda x: x.add(10))

Utils
=====

make\_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale\_each=False, pad\_value=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a 4D mini-batch Tensor of shape (B x C x H x W),
or a list of images all of the same size,
makes a grid of images

normalize=True will shift the image to the range (0, 1),
by subtracting the minimum and dividing by the maximum pixel value.

if range=(min, max) where min and max are numbers, then these numbers are used to
normalize the image.

scale_each=True will scale each image in the batch of images separately rather than
computing the (min, max) over all images.

pad_value=<float> sets the value for the padded pixels.

`Example usage is given in this notebook` <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>

save\_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale\_each=False, pad\_value=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saves a given Tensor into an image file.

If given a mini-batch tensor, will save the tensor as a grid of images.

All options after `filename` are passed through to `make_grid`. Refer to it's documentation for
more details
