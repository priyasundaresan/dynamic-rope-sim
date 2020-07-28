# Keypoint detection
Forked from https://github.com/hackiey/keypoints
Based on "Towards Accurate Multi-person Pose Estimation in the Wild" from Google: https://arxiv.org/pdf/1701.01779.pdf

# Preparing dataset
Dataset organization:
  ~~~
  dataset_name
  `-- train
  |   `-- images
  |   |   `-- |-- 00000.jpg
  |   |       |-- 00001.jpg
  |   |       |-- ...
  |   |       `-- ...
  |   `-- keypoints # annotations  [x1,y1,x2,y2,x3,y3,...] where (xk,yk) is one keypoint annotation
  |       `-- |-- 00000.npy
  |           |-- 00001.npy
  |           |-- ...
  |           `-- ...
  `-- test
      `-- images
      |   `-- |-- 00000.jpg
      |       |-- 00001.jpg
      |       |-- ...
      |       `-- ...
      `-- keypoints # annotations  [x1,y1,x2,y2,x3,y3,...] where (xk,yk) is one keypoint annotation
          `-- |-- 00000.npy
              |-- 00001.npy
              |-- ...
              `-- ...
  ~~~
Make a directory called `data` at the same level as `src` and move `dataset_name` inside that

# Training
Configure `config.py` and `train.py` with paths to data and output directory
`python train.py` 

# Evaluation
`python analysis.py`
