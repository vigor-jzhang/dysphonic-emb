import random
import torch
import torchaudio
from hparam import hparam as hp
import numpy as np
import augment
import glob
import librosa
import math

def form_list(fpath):
	fh = open(fpath)
	templist = fh.readlines()
	fh.close()
	flist = []
	for item in templist:
		flist.append(item.rstrip('\n'))
	return flist

def get_flist(data_path):
	# Form normal male list
	fpath = data_path+'n_m.txt'
	n_m_list = form_list(fpath)
	# Form normal female list
	fpath = data_path+'n_w.txt'
	n_w_list = form_list(fpath)
	# Form patho male list
	fpath = data_path+'p_m.txt'
	p_m_list = form_list(fpath)
	# Form patho female list
	fpath = data_path+'p_w.txt'
	p_w_list = form_list(fpath)
	return n_m_list, n_w_list, p_m_list, p_w_list

def IR_aug(audio,sr):
	ir_list = glob.glob('../IR_aug_25k/*.npy')
	random.shuffle(ir_list)
	h = np.load(ir_list[0])
	# change audio to numpy array
	audio = audio.squeeze(0).numpy()
	N = audio.shape[0]
	# convolve the impulse response
	audio_conv = np.convolve(audio,h)
	# downsample the convolved audio
	audio = audio_conv[:N]
	audio = torch.from_numpy(np.float32(audio)).unsqueeze(0)
	return audio

def aug_part(audio, sr):
	# Define random effect application
	flag = [random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)]
	if flag[0]==1:
		# Impulse response augmentation
		audio = IR_aug(audio,sr)
	if flag[1]==1:
		audio = 0.9*audio/torch.max(torch.abs(audio))
		noise_level = random.random()*0.005
		noise = torch.randn(audio.shape)*noise_level
		audio = audio+noise
	if flag[2]==1:
		bgn_list = glob.glob('../noise_wav/*.wav')
		random.shuffle(bgn_list)
		bgn = torchaudio.load(bgn_list[0])
		bgn = bgn[0]
		start = random.randint(0,bgn.shape[1]-audio.shape[1])
		end = start+audio.shape[1]
		bgn = bgn[0,start:end]
		noise_level = random.random()*0.01
		audio = audio+bgn*noise_level
	if flag[3]==1:
		src_info = {'rate': sr}
		target_info = {
			'channels': 1,
			'length': 0,
			'rate': sr,
		}
		random_pitch = lambda: np.random.randint(-200, 200)
		audio = augment.EffectChain().pitch(random_pitch).rate(sr).apply(\
			audio, src_info=src_info, target_info=target_info)
	return audio

def gen_spec_from_wav(f):
	audio, sr = torchaudio.load(f)
	if hp.data.sr != sr:
		raise ValueError(f'Invalid sample rate {sr}.')
	audio = 0.9*audio/torch.max(torch.abs(audio))
	audio = aug_part(audio,sr)
	clip_level = random.random()/2+0.4
	audio = clip_level*audio/torch.max(torch.abs(audio))
	mel_args = {
		'win_length': hp.data.hop_size * 4,
		'hop_length': hp.data.hop_size,
		'n_fft': hp.data.n_fft,
	}
	spec_transform = torchaudio.transforms.Spectrogram(**mel_args)
	with torch.no_grad():
		spec = spec_transform(audio)
		spec = 20 * torch.log10(torch.clamp(spec, min=1e-5)) - 20
	return spec.permute(0,2,1).squeeze(0)

def spec_norm(spec):
	with torch.no_grad():
		spec_shape = spec.shape
		spec = spec.view(spec.size(0), -1)
		spec = spec-spec.min(1, keepdim=True)[0]
		spec = spec/spec.max(1, keepdim=True)[0]
		spec = spec.view(spec_shape)
	return spec

def get_spec(npylist, n, now_tag):
	temp_spec_list = []
	counter = 0
	for item in npylist:
		try:
			spec = gen_spec_from_wav(item[:-7]+now_tag+'.wav')
		except:
			continue
		# Then compare length
		if spec.shape[0]<hp.data.frames+1:
			continue
		# Random get part of spec
		start = random.randint(0, spec.shape[0] - hp.data.frames)
		end = start + hp.data.frames
		spec = spec[start:end]
		spec = spec_norm(spec)
		if torch.isnan(spec).any():
			continue
		# Finish random masking
		temp_spec_list.append(spec)
		counter += 1
		if counter>n-1:
			break
	# Stack over
	stacked_spec = torch.stack(temp_spec_list)
	return stacked_spec

class EmbSEDataset(torch.utils.data.Dataset):
	def __init__(self, data_path):
		super().__init__()
		n_m_list, n_w_list, p_m_list, p_w_list = get_flist(data_path)
		self.n_m_list = n_m_list
		self.p_m_list = p_m_list
		self.n_w_list = n_w_list
		self.p_w_list = p_w_list
		# Get label
		label = torch.zeros(2*hp.train.wav_n).long()
		label[hp.train.wav_n:] = label[hp.train.wav_n:]+1
		self.label = label

	def __len__(self):
		#return int(len(self.n_w_list)+len(self.n_m_list))
		return int(len(self.n_w_list))

	def __getitem__(self, idx):
		#tag_list = ['a_n','a_l','a_h','i_n','i_l','i_h','u_n','u_l','u_h']
		#random.shuffle(tag_list)
		#now_tag = tag_list[0]
		now_tag = 'a_n'
		# Male or female depend on odd or even
		if idx%2==0:
			random.shuffle(self.n_m_list)
			random.shuffle(self.p_m_list)
			norm_spec = get_spec(self.n_m_list,hp.train.wav_n,now_tag)
			patho_spec = get_spec(self.p_m_list,hp.train.wav_n,now_tag)
		else:
			random.shuffle(self.n_w_list)
			random.shuffle(self.p_w_list)
			norm_spec = get_spec(self.n_w_list,hp.train.wav_n,now_tag)
			patho_spec = get_spec(self.p_w_list,hp.train.wav_n,now_tag)
		# Stack spec
		#spec = torch.cat([norm_spec,patho_spec],dim=0)
		return {
			'norm_spec': norm_spec,
			'patho_spec': patho_spec,
			'label': self.label,
		}

def emb_dataloader(data_path):
	dataloader = EmbSEDataset(data_path)
	return torch.utils.data.DataLoader(
		dataloader,
		batch_size=hp.train.batch_size,
		collate_fn=None,
		shuffle=False,
		num_workers=hp.train.num_workers,
		pin_memory=True,
		drop_last=True)

if __name__ == '__main__':
	dataloader = auto_dataloader(hp.train.train_path)
	for inputs in dataloader:
		print(inputs.shape)