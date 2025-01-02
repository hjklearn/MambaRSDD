import torch as t
import torch.nn.functional as F
import torch
import sys
sys.path.append('/root/autodl-tmp/MambaRSDD/MambaRSDD/')
from torch import nn
from rail_defect import RDDataset
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import  numpy as np
from datetime import datetime
rootpath= '/root/autodl-tmp/RGB_D_rail/RGB_D_rail/'
TestDatasets = RDDataset(rootpath, 'test')
# test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=4)
test_dataloader = DataLoader(TestDatasets, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader2 = DataLoader(testData2, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader3 = DataLoader(testData3, batch_size=1, shuffle=False, num_workers=4)

from mamba_rail_net import Depth_Mamba               # student

net = Depth_Mamba()
net.load_state_dict(t.load('/root/autodl-tmp/MambaRSDD/MambaRSDD/RSDD_Tool/Pth/MambaRSDD_×DA_√FF_√CL_√F_path2024_12_27_20_34_best.pth'))   ########gaiyixia
a = '/root/autodl-tmp/MambaRSDD/MambaRSDD/RSDD_Tool/消融实验/'
b = '/×DA_√FF_√CL_√F/'                                                                               
c = '/rail_362/'
d = '/VT1000/'
e = '/VT5000/'

aa = []

vt800 = a + b + c
vt1000 = a + b + d
vt5000 = a + b + e

path1 = vt800
isExist = os.path.exists(vt800)
if not isExist:
	os.makedirs(vt800)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0

	for i, sample in enumerate(test_dataloader):

		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()

		sal_finally, rgb_list, dep_list = net(image, depth)

		# dep1 = F.interpolate(input=dep1, scale_factor=4, mode='bilinear', align_corners=True)
		# dep1, _ = torch.max(dep1, dim=1)
		# out = torch.sigmoid(dep1.unsqueeze(1))
		out = torch.sigmoid(sal_finally)

		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()

		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')
		print(path1 + name + '.png')

