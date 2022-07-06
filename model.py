import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from hparam import hparam as hp
from math import sqrt
from utils import get_centroids, get_cossim, calc_loss

def Conv2d(*args, **kwargs):
	layer = nn.Conv2d(*args, **kwargs)
	nn.init.kaiming_normal_(layer.weight)
	return layer

def Linear(*args, **kwargs):
	layer = nn.Linear(*args, **kwargs)
	nn.init.kaiming_normal_(layer.weight)
	return layer

class ResidualBlock(nn.Module):
	def __init__(self, resN, dilation):
		super().__init__()
		self.dilated_conv = Conv2d(resN, 2 * resN, 3, padding=dilation, dilation=dilation)
		self.output_projection = Conv2d(resN, resN, 3, padding=dilation, dilation=dilation)

	def forward(self, x):
		y = self.dilated_conv(x)
		gate, filter = torch.chunk(y, 2, dim=1)
		y = torch.sigmoid(gate) * torch.tanh(filter)

		y = self.output_projection(y)
		return (x + y) / sqrt(2.0)


class SpeechEmbedder(nn.Module):
	def __init__(self):
		super().__init__()
		self.in_proj = nn.Sequential(
			Conv2d(1, hp.model.res_map_n, 3, padding=1),
		)
		self.resnet = nn.ModuleList([
			ResidualBlock(hp.model.res_map_n, 2**(i // hp.model.dilation_cycle))
			for i in range(hp.model.residual_layers)
		])
		self.out_conv = nn.Sequential(
			Conv2d(hp.model.res_map_n, 1, 3, padding=1),
			nn.LeakyReLU(0.4),
		)
		self.out_linear_1 = nn.Sequential(
			nn.Linear(int(hp.data.n_fft/2)+1,hp.model.proj),
			nn.LeakyReLU(0.4),
		)
		self.out_linear_2 = nn.Sequential(
			nn.Linear(hp.model.proj*hp.data.frames,hp.model.proj),
		)

	def forward(self, x):
		x = self.in_proj(x.unsqueeze(1))
		for layer in self.resnet:
			x = layer(x)
		x = self.out_conv(x)
		x = self.out_linear_1(x.squeeze(1))
		x = self.out_linear_2(x.reshape(x.shape[0],-1))
		return x

class SpeechClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.embedder = SpeechEmbedder()
		self.acti_proj = nn.Sequential(
			nn.ReLU(),
			Linear(hp.model.proj,int(hp.model.proj/2)),
			nn.ReLU(),
			Linear(int(hp.model.proj/2),hp.model.class_n),
			nn.LogSoftmax(dim=1),
		)

	def forward(self,x):
		x = self.embedder(x)
		x = self.acti_proj(x)
		return x

class GE2ELoss(nn.Module):
	def __init__(self, device):
		super(GE2ELoss, self).__init__()
		self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
		self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
		self.device = device

	def forward(self, embeddings):
		torch.clamp(self.w, 1e-6)
		centroids = get_centroids(embeddings)
		cossim = get_cossim(embeddings, centroids)
		sim_matrix = self.w*cossim.to(self.device) + self.b
		loss, _ = calc_loss(sim_matrix)
		return loss

if __name__ == '__main__':
	model = SpeechClassifier()
	mel = torch.rand(3,30,513)
	res = model(mel)
	print(res.shape)
