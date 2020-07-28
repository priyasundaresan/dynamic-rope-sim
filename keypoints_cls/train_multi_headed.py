import pickle
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
from src.model_multi_headed import KeypointsGauss
from src.dataset_multi_headed import KeypointsDataset, transform

MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss()
crossEntropyLoss = F.cross_entropy

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def forward(sample_batched, model):
    img, gt_gauss, cls = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_gauss, cls_pred = model.forward(img)
    cls_loss = crossEntropyLoss(cls_pred, cls.cuda().long()).double()
    kpt_loss = bceLoss(pred_gauss.double(), gt_gauss)
    cls_correct = torch.argmax(cls_pred).item() == cls.item()
    return cls_loss, kpt_loss, cls_correct

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    train_losses = []
    test_losses = []
    train_cls_losses = []
    test_cls_losses = []
    train_kpts_losses = []
    test_kpts_losses = []
    train_cls_acc = []
    test_cls_acc = []
    for epoch in range(epochs):

        train_loss = 0.0
        train_kpt_loss = 0.0
        train_cls_loss = 0.0
        correct = 0
        seen = 0
        for i_batch, sample_batched in enumerate(train_data):
            #if i_batch>10:
            #    break
            optimizer_kpt.zero_grad()
            optimizer_cls.zero_grad()
            cls_loss, kpt_loss, cls_correct = forward(sample_batched, model)
            kpt_loss.backward(retain_graph=True)
            cls_loss.backward(retain_graph=True)
            optimizer_kpt.step()
            optimizer_cls.step()
            train_loss += kpt_loss.item() + cls_loss.item()
            train_kpt_loss += kpt_loss.item()
            train_cls_loss += cls_loss.item()
            correct += cls_correct 
            accuracy = 0 if not seen else correct/seen
            print('[%d, %5d] kpts loss: %.3f, cls loss: %.3f, cls_accuracy: %.3f' % \
	           (epoch + 1, i_batch + 1, kpt_loss.item(), cls_loss.item(), accuracy), end='')
            print('\r', end='')
            seen += 1
        train_kpts_losses.append(train_kpt_loss/i_batch)
        train_cls_losses.append(train_cls_loss/i_batch)
        train_losses.append(train_loss/i_batch)
        train_cls_acc.append(accuracy)
        print('train loss:', train_loss/i_batch)
        print('train kpt loss:', train_kpt_loss/i_batch)
        print('train cls loss:', train_cls_loss/i_batch)
        print('train cls_accuracy:', accuracy)
        
        test_loss = 0.0
        test_kpt_loss = 0.0
        test_cls_loss = 0.0
        correct = 0
        seen = 0
        for i_batch, sample_batched in enumerate(test_data):
            #if i_batch>10:
            #    break
            cls_loss, kpt_loss, cls_correct = forward(sample_batched, model)
            correct += cls_correct 
            accuracy = 0 if not seen else correct/seen
            test_loss += kpt_loss.item() + cls_loss.item()
            test_kpt_loss += kpt_loss.item()
            test_cls_loss += cls_loss.item()
            seen += 1
        test_kpts_losses.append(test_kpt_loss/i_batch)
        test_cls_losses.append(test_cls_loss/i_batch)
        test_losses.append(test_loss/i_batch)
        test_cls_acc.append(accuracy)
        print('test loss:', test_loss/i_batch)
        print('test kpt loss:', test_kpt_loss/i_batch)
        print('test cls loss:', test_cls_loss/i_batch)
        print('test cls_accuracy:', accuracy)
        torch.save(keypoints.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '.pth')

    history =  {"train_losses": train_losses, \
		"train_kpt_losses": train_kpts_losses, \
		"train_cls_losses": train_cls_losses, \
	    	"train_cls_accs": train_cls_acc, \
		"test_losses": test_losses, \
		"test_kpt_losses": test_kpts_losses, \
		"test_cls_losses": test_cls_losses, \
		"test_cls_accs": test_cls_acc}
    with open('%s/history.pickle'%checkpoint_path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #pprint.pprint(history) # DEBUG
    return history

# dataset
workers=0
dataset_dir = 'undo_reid_term'
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, dataset_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = KeypointsDataset('data/%s/train/images'%dataset_dir,
                           'data/%s/train/actions'%dataset_dir, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = KeypointsDataset('data/%s/test/images'%dataset_dir,
                           'data/%s/test/actions'%dataset_dir, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH).cuda()

# optimizer
optimizer_kpt = optim.Adam(keypoints.parameters(), lr=1e-4, weight_decay=1.0e-4)
optimizer_cls = optim.Adam(keypoints.parameters(), lr=1e-4, weight_decay=1.0e-4)

history = fit(train_data, test_data, keypoints, epochs=epochs, checkpoint_path=save_dir)
plot_history(history, epochs, save_dir)
