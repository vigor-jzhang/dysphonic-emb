import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import random

from hparam import hparam as hp
from dataloader import emb_dataloader
from model import SpeechClassifier, GE2ELoss

def create_model(ckpt_path):
	# Define device
	device = torch.device(hp.device)
	# Define classifier model and loss module
	classifier = SpeechClassifier().to(device)
	ge2e_loss = GE2ELoss(device)
	# Define optimizer
	opt_fn = torch.optim.SGD([
		{'params': classifier.parameters()},
		{'params': ge2e_loss.parameters()}
	], lr=hp.train.lr, weight_decay=5e-5)
	torch.save({
		'epoch': 0,
		'classifier': classifier.state_dict(),
		'ge2e': ge2e_loss.state_dict(),
		'optimizer': opt_fn.state_dict(),
		},ckpt_path)

def train(previous_ckpt_path, writer):
	# Define device
	device = torch.device(hp.device)
	# Define dataloader
	dataloader = emb_dataloader(hp.train.train_path)
	# Define classifier model and loss module
	classifier = SpeechClassifier().to(device)
	ge2e_loss = GE2ELoss(device)
	class_loss = nn.NLLLoss()
	# Load pretrained parameters
	checkpoint = torch.load(previous_ckpt_path)
	classifier.load_state_dict(checkpoint['classifier'])
	ge2e_loss.load_state_dict(checkpoint['ge2e'])
	# Define optimizer
	opt_fn = torch.optim.SGD([
		{'params': classifier.parameters()},
		{'params': ge2e_loss.parameters()}
	], lr=hp.train.lr, weight_decay=5e-5)
	opt_fn.load_state_dict(checkpoint['optimizer'])
	# Set to train mode
	classifier.train()
	start_epoch = checkpoint['epoch']
	# Not sure why this is important
	torch.backends.cudnn.benchmark = True
	# Start training
	for epoch in range(start_epoch,start_epoch+hp.train.ckpt_interval):
		total_loss = 0
		for inputs in dataloader:
			# Get data
			norm_spec = inputs['norm_spec'].squeeze(0).to(device)
			patho_spec = inputs['patho_spec'].squeeze(0).to(device)
			spec = torch.cat([norm_spec,patho_spec],dim=0)
			label = inputs['label'].squeeze(0).to(device)
			# random arrange
			perm = random.sample(range(0, hp.train.wav_n*2), hp.train.wav_n*2)
			unperm = list(perm)
			for i,j in enumerate(perm):
				unperm[j] = i
			spec = spec[perm]
			label = label[perm]
			# Train network
			opt_fn.zero_grad()
			emb = classifier.embedder(spec)
			res = classifier.acti_proj(emb)
			emb = emb[unperm]
			emb = torch.reshape(emb, (2, hp.train.wav_n, emb.size(1)))
			# Get loss, call backward, step optimizer
			temp_ge2e_loss = ge2e_loss(emb) # wants (normal or pathology, Utterances, embedding)
			temp_class_loss = class_loss(res,label)
			loss = (1.0*temp_class_loss)+(1.0*0.035*temp_ge2e_loss)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(classifier.parameters(), 3.0)
			torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
			opt_fn.step()
			total_loss = total_loss+loss
		# Wrtie total loss to log
		print('Epoch [{}] - NLL+GE2E loss: {}'.format(epoch+1,total_loss))
		writer.add_scalar('Trian total loss', total_loss, epoch+1)
	# Save checkpoint
	checkpoint_path = hp.train.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(epoch+1)
	print('Saving model in ['+checkpoint_path+']')
	torch.save({
		'epoch': epoch+1,
		'classifier': classifier.state_dict(),
		'ge2e': ge2e_loss.state_dict(),
		'optimizer': opt_fn.state_dict(),
		},checkpoint_path)
	print('Saving END')
	return checkpoint_path

if __name__ == '__main__':
	# Write tensorboard
	torch.manual_seed(233)
	os.makedirs(hp.train.ckpt_dir, exist_ok=True)
	writer = SummaryWriter(hp.train.ckpt_dir)
	# Restore ckpt or create new
	if (not hp.train.restoring):
		create_model(hp.train.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(0))
		previous_ckpt = hp.train.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(0)
	else:
		previous_ckpt = hp.train.restore_path
	# Load epochs
	checkpoint = torch.load(previous_ckpt)
	start_epoch = checkpoint['epoch']
	# Training procedure
	for epoch in range(start_epoch,hp.train.epochs,hp.train.ckpt_interval):
		# Training
		now_ckpt = train(previous_ckpt, writer)
		previous_ckpt = now_ckpt

