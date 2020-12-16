import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import pdb
import math
import sys
sys.dont_write_bytecode = True



###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
	print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer



def define_Net(pretrained=False, model_root=None, which_model='Conv64', metric='ImgtoClass', norm='batch', init_type='normal', use_gpu=True, **kwargs):
	Net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if which_model == 'Conv64F' and metric == 'ImgtoClass':
		Net = ImgtoClass_64F(norm_layer=norm_layer, **kwargs)
	elif which_model == 'Conv64F' and metric == 'Prototype':
		Net = Prototype_64F(norm_layer=norm_layer, **kwargs)
	elif which_model == 'ResNet256F':
		net_opt = {'userelu': False, 'in_planes':3, 'dropout':0.5, 'norm_layer': norm_layer} 
		Net = ResNetLike(net_opt)
	else:
		raise NotImplementedError('Model name [%s] is not recognized' % which_model)
	init_weights(Net, init_type=init_type)

	if use_gpu:
		Net.cuda()

	if pretrained:
		Net.load_state_dict(model_root)

	return Net


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)



##############################################################################
# Classes: FourLayer_64F
##############################################################################

# Model: FourLayer_64F 
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer  
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class ImgtoClass_64F(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3, semi_neighbor_k=3):
		super(ImgtoClass_64F, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                              # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21
		)
		
		self.metric = ImgtoClass_Metric(neighbor_k=neighbor_k)  # 1*num_classes
		self.aug = Support_AUG_Image(neighbor_k=semi_neighbor_k)
		# self.aug = Support_AUG(neighbor_k=semi_neighbor_k)



	def forward(self, input1, input2, input3):

		# extract features of input1--query image
		q = self.features(input1)

		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			support_set_sam = self.features(input2[i])
			B, C, h, w = support_set_sam.size()
			support_set_sam = support_set_sam.permute(1, 0, 2, 3)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			S.append(support_set_sam)

		u = self.features(input3)
		S = self.aug(u, S)

		x = self.metric(q, S) # get Batch*num_classes

		return x



#========================== Define an image-to-class layer ==========================#


class ImgtoClass_Metric(nn.Module):
	def __init__(self, neighbor_k=3):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k


	# Calculate the k-Nearest Neighbor of each local descriptor 
	def cal_cosinesimilarity(self, input1, input2):
		B, C, h, w = input1.size()
		Similarity_list = []

		for i in range(B):
			query_sam = input1[i]
			query_sam = query_sam.view(C, -1)
			query_sam = torch.transpose(query_sam, 0, 1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)   
			query_sam = query_sam/query_sam_norm

			if torch.cuda.is_available():
				inner_sim = torch.zeros(1, len(input2)).cuda()

			for j in range(len(input2)):
				support_set_sam = input2[j]
				support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
				support_set_sam = support_set_sam/support_set_sam_norm

				# cosine similarity between a query sample and a support category
				innerproduct_matrix = query_sam@support_set_sam

				# choose the top-k nearest neighbors
				topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
				inner_sim[0, j] = torch.sum(topk_value)
				# inner_sim[0, j] = torch.sum(innerproduct_matrix)

			Similarity_list.append(inner_sim)

		Similarity_list = torch.cat(Similarity_list, 0)    

		return Similarity_list 


	def forward(self, x1, x2):

		Similarity_list = self.cal_cosinesimilarity(x1, x2)

		return Similarity_list


class Support_AUG_Image(nn.Module):
	def __init__(self, neighbor_k=3):
		super(Support_AUG_Image, self).__init__()
		self.neighbor_k = neighbor_k
	
	def knn(self, input1, input2):
		B, C, h, w = input1.size()
		Similarity_list = []

		for i in range(B):
			query_sam = input1[i]
			query_sam = query_sam.view(C, -1)
			query_sam = torch.transpose(query_sam, 0, 1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)   
			query_sam = query_sam/query_sam_norm

			if torch.cuda.is_available():
				inner_sim = torch.zeros(1, len(input2)).cuda()

			for j in range(len(input2)):
				support_set_sam = input2[j]
				support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
				support_set_sam = support_set_sam/support_set_sam_norm

				# cosine similarity between a query sample and a support category
				innerproduct_matrix = query_sam@support_set_sam

				# choose the top-k nearest neighbors
				topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
				inner_sim[0, j] = torch.sum(topk_value)
				# inner_sim[0, j] = torch.sum(innerproduct_matrix)

			Similarity_list.append(inner_sim)

		Similarity_list = torch.cat(Similarity_list, 0)
		Similarity_list = F.softmax(Similarity_list, 1)
		
		for j in range(len(input2)):
			select_num = 10
			_, select_index = torch.topk(Similarity_list[:, j], select_num)
			selected = input1[select_index, :, :, :]
			selected = selected.permute(1, 0, 2, 3)
			selected = selected.contiguous().view(C, -1)
			input2[j] = torch.cat((input2[j], selected), 1)

		return input2

	def forward(self, x1, x2):
		return self.knn(x1, x2)


class Support_AUG(nn.Module):
	def __init__(self, neighbor_k=3, ratio=0.02):
		super(Support_AUG, self).__init__()
		self.neighbor_k = neighbor_k
		self.select_ratio = ratio
	
	def knn(self, input1, input2):
		B, C, h, w = input1.size()

		unlabel_sam_raw = input1.permute(1, 0, 2, 3)
		unlabel_sam_raw = unlabel_sam_raw.contiguous().view(C, -1)
		unlabel_sam = torch.transpose(unlabel_sam_raw, 0, 1)
		unlabel_sam_norm = torch.norm(unlabel_sam, 2, 1, True)   
		unlabel_sam = unlabel_sam/unlabel_sam_norm

		for j in range(len(input2)):
			support_set_sam = input2[j]
			support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
			support_set_sam = support_set_sam/support_set_sam_norm

			innerproduct_matrix = unlabel_sam@support_set_sam

			topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
			patch_to_class_sim = torch.sum(topk_value, 1)
			select_num = int(patch_to_class_sim.shape[0]*self.select_ratio)
			_, select_index = torch.topk(patch_to_class_sim, select_num)
			
			input2[j] = torch.cat((input2[j], unlabel_sam_raw[:,select_index]), 1)

		# patch_to_class_sim = []
		# for j in range(len(input2)):
		# 	support_set_sam = input2[j]
		# 	support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
		# 	support_set_sam = support_set_sam/support_set_sam_norm

		# 	innerproduct_matrix = unlabel_sam@support_set_sam

		# 	topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
		# 	patch_to_class_sim.append(torch.sum(topk_value, 1))
			
		# patch_to_class_sim = [x.unsqueeze(0) for x in patch_to_class_sim]
		# patch_to_class_sim = torch.cat(patch_to_class_sim, 0)
		# patch_to_class_sim_sum = torch.sum(patch_to_class_sim, 0)
		# patch_to_class_sim_weight = patch_to_class_sim/patch_to_class_sim_sum
		# patch_to_class_sim.mul(patch_to_class_sim_weight)
		
		# for j in range(len(input2)):
		# 	select_num = int(patch_to_class_sim[j].shape[0]*self.select_ratio)
		# 	_, select_index = torch.topk(patch_to_class_sim[j], select_num)
		# 	input2[j] = torch.cat((input2[j], unlabel_sam_raw[:,select_index]), 1)

		return input2

	def forward(self, x1, x2):
		return self.knn(x1, x2)



class Prototype_64F(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3, semi_neighbor_k=3):
		super(Prototype_64F, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                              # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                                # 64*21*21
		)
		
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.metric = Proto_Metric()  # 1*num_classes
		self.metric_semi = Proto_Metric_Semi()
		# self.att = eca_layer(channel=64)

	def aug(self, x1, x2):
			B, _, _, _ = x1.size()
			confidence = F.softmax(self.metric(x1, x2), 1)

			for j in range(len(x2)):
					select_num = int(B * 0.1)
					_, select_index = torch.topk(confidence[:, j], select_num)
					selected = x1[select_index, :, :, :]
					x2[j] = torch.cat((x2[j], selected), 0)

			return x2

	def forward(self, input1, input2, input3):
		# extract features of input1--query image
		q = self.features(input1)
		# q = self.avgpool(self.att(q))
		q = self.avgpool(q)

		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			support_set_sam = self.features(input2[i])
			# support_set_sam = self.avgpool(self.att(support_set_sam))
			support_set_sam = self.avgpool(support_set_sam)
			S.append(support_set_sam)

		u = self.features(input3)
		u = self.avgpool(u)
		w = F.softmax(self.metric(u, S), 1)

		x = self.metric_semi(q, S, u, w) # get Batch*num_classes

		return x


class Proto_Metric_Semi(nn.Module):
	def __init__(self):
		super(Proto_Metric_Semi, self).__init__()

	def proto(self, input1, input2, input3, weight):
			B, c, _, _ = input1.size()
			Distance_list = []

			for i in range(B):
					query_sam = input1[i].squeeze()
					unlabel_sam = input3.squeeze()

					if torch.cuda.is_available():
							score = torch.zeros(1, len(input2)).cuda()

					for j in range(len(input2)):
							support_set_sam = input2[j].squeeze()
							soft_weight = weight[:, j]
							support_set_sam = (torch.sum(support_set_sam, dim=0) + torch.sum(soft_weight.unsqueeze(1)*unlabel_sam, dim=0))/((input2[j].size()[0]) + torch.sum(soft_weight))

							dist_matrix = -torch.sum((query_sam-support_set_sam)**2, 0)
							score[0, j] = dist_matrix

					Distance_list.append(score)

			Distance_list = torch.cat(Distance_list, 0)

			return Distance_list

	
	def forward(self, x1, x2, x3, w):
		return self.proto(x1, x2, x3, w)


class Proto_Metric(nn.Module):
	def __init__(self):
		super(Proto_Metric, self).__init__()

	def proto(self, input1, input2):
			B, c, _, _ = input1.size()
			Distance_list = []

			for i in range(B):
					query_sam = input1[i].squeeze()

					if torch.cuda.is_available():
							score = torch.zeros(1, len(input2)).cuda()

					for j in range(len(input2)):
							support_set_sam = input2[j].squeeze()
							support_set_sam = torch.sum(support_set_sam, dim=0)/(input2[j].size()[0])

							dist_matrix = -torch.sum((query_sam-support_set_sam)**2, 0)
							score[0, j] = dist_matrix

					Distance_list.append(score)

			Distance_list = torch.cat(Distance_list, 0)

			return Distance_list

	
	def forward(self, x1, x2):
		return self.proto(x1, x2)


class eca_layer(nn.Module):
        """Constructs a ECA module.
        Args:
                channel: Number of channels of the input feature map
                k_size: Adaptive selection of kernel size
        """
        def __init__(self, channel, k_size=3):
                super(eca_layer, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
                self.sigmoid = nn.Sigmoid()

        def forward(self, x):
                # x: input features with shape [b, c, h, w]
                b, c, h, w = x.size()

                # feature descriptor on the global spatial information
                y = self.avg_pool(x)

                # Two different branches of ECA module
                y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

                # Multi-scale information fusion
                y = self.sigmoid(y)

                return x * y.expand_as(x)
