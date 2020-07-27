import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/host/src')

class KeypointsGauss(nn.Module):
	def __init__(self, num_keypoints, img_height=480, img_width=640, num_classes=3):
		super(KeypointsGauss, self).__init__()
		self.num_keypoints = num_keypoints
		self.img_height = img_height
		self.num_classes = num_classes
		self.img_width = img_width
		self.resnet = torchvision.models.resnet18()
		self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
		self.conv1by1 = nn.Conv2d(512, self.num_keypoints, (1,1))
		self.dropout_1 = nn.Dropout2d(p=0.5)
		self.conv_transpose = nn.ConvTranspose2d(self.num_keypoints, self.num_keypoints, kernel_size=32, stride=8)
		self.avg_pool = nn.AvgPool2d(kernel_size=8)
		self.linear = nn.Linear(in_features=1024, out_features=self.num_classes, bias=True)
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
		x = self.resnet(x) 
		class_scores = self.avg_pool(x)
		class_scores = class_scores.view(class_scores.shape[0], 1024)
		class_scores = self.linear(class_scores)
		#class_scores = F.softmax(class_scores)
		x = self.conv1by1(x)
		x = self.dropout_1(x)
		x = self.conv_transpose(x)
		output = nn.Upsample(size=(self.img_height, self.img_width), mode='bilinear')(x)
		heatmaps = self.sigmoid(output[:,:self.num_keypoints, :, :])
		return heatmaps, class_scores

if __name__ == '__main__':
	model = KeypointsGauss(4).cuda()
	x = torch.rand((1,3,480,640)).cuda()
	result = model.forward(x)
	print(x.shape)
	print(result.shape)
