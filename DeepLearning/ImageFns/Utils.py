import matplotlib.pyplot as plt
import numpy as np
import os
import json
import struct
import random

import torch

from skimage.metrics import structural_similarity as ssim
from PIL import ImageDraw

# from tensorflow.keras.applications.resnet_v2 import decode_predictions

# imagenet_decoder = decode_predictions


def tensorToNPImage(tensor):
	numpy = tensor.permute(0,2,3,1).detach().cpu().numpy()

	return numpy



def binary(num):
	multiplier = pow(2, 9)
	packed = struct.pack('!h', int(num * multiplier))
	integers = [c for c in packed]
	binaries = [bin(i) for i in integers]
	stripped_binaries = [s.replace('0b', '') for s in binaries]
	padded = [s.rjust(8, '0') for s in stripped_binaries]
	return ''.join(padded)


def print_out(name, kern, file_name='./matrix.h', f1=None):
	# [batches,cols,rows,chans]
	comma = True
	if (f1 == None):
		f1 = open(file_name, 'w+')
	
	batch_size = kern.shape[0]
	cols = kern.shape[1]
	rows = kern.shape[2]
	chans = kern.shape[3]
	
	f1.write("assign ")
	f1.write(name)
	f1.write(" = { ")
	
	for k in reversed(range(0, chans)):
		for j in reversed(range(0, cols)):
			for i in reversed(range(0, rows)):
				if not (comma):
					f1.write(" , ")
				else:
					comma = False
				f1.write("16'b")
				f1.write(binary(kern[0, j, i, k]))
	
	f1.write(" };\n\n")


def print_image_values(imgs, name = "python_values", file_path = None):
	if file_path is None:
		file_path = "./Outputs/modelWeights/CNN1C2D/images/"
	
	for i in range(imgs.shape[0]):
		file_name = file_path + name +".vh"
		f_weights = open(file_name, 'w+')
		print_out(name, imgs, f1=f_weights)


def imagenet_decoder(preds):
	out_shape = preds.shape
	IN_label_path = './ImageDatasets/imagenet21k_resized/class_index.json'
	IN_label = json.load(open(IN_label_path))
	preds = np.reshape(preds,(-1,))
	
	if type(preds[0]) is type(np.asarray(["s"])[0]) :
		labels = [IN_label[p] for p in preds]
		labels = np.reshape(labels,out_shape)
		
		return labels
	else:
		labels = [list(IN_label.keys())[p] for p in preds]
		# [IN_label[p] for p in preds]
		labels = np.reshape(labels,out_shape)
		
		return labels
	
	

def normalize(imgs):
	mi = np.min(imgs, (1, 2, 3), keepdims=True)
	ma = np.max(imgs, (1, 2, 3), keepdims=True)
	imgs = (imgs - mi + 0.000003)/(ma - mi + 0.000003)
	return imgs

def make_ssim(x, y, fh=5, get_mean=False, normal=True):
	t = ssim(x, y, data_range=1.0, win_size=fh, multichannel=True, full=True)
	
	means = t[0]
	imgs = t[1]
	
	if normal:
		imgs = normalize(imgs)
	
	if get_mean:
		return imgs, means
	else:
		return imgs

def plot_grid(imgs, lbls=None, preds=None, cols=10, nmax=30, save_path=None, pause=0.1):
	"""
	  This function plots images.

	  Inputs:
		  imgs - A numpy array of the images to be plotted
		lbls(optional) - ground truth labels of imgs
		preds(optional) - predicted labels of imgs
		cols(default: 13) - the number of images to be plotted on each row
		img_sie(default: (28,28,1)) - the size of each element in imgs
		nmax(default: 65) - a limit on the number of imgs to print (the default prints 5 rows of 13)

	"""
	if type(imgs) == torch.Tensor:
		imgs = tensorToNPImage(imgs)

	imgsize = imgs.shape[1:]
	subplot_row = 0
	
	img_count = min(imgs.shape[0], nmax)
	rows = int(-(-img_count // cols))
	extras = img_count % cols
	
	img_width = min(2 * cols, 26)
	img_height = img_width/cols*1.25*rows
	
	# _plt.clf( )
	fig, axes = plt.subplots(rows, cols,
							 figsize=(img_width, img_height), squeeze=False,
							 subplot_kw=dict(aspect='equal'))
	
	for i in range(img_count):
		subplot_row = i//cols
		subplot_col = i % cols
		
		ax = axes[subplot_row, subplot_col]
		
		if imgsize[2] == 1:
			# cm = 'Greys_r'
			plottable_image = np.reshape(imgs[i], imgsize[0:2])
		else:
			# cm = None
			plottable_image = np.reshape(imgs[i], imgsize)
		
		fig = ax.imshow(plottable_image, vmin=0.0, vmax=1.0, cmap='Greys_r')
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		
		if preds is None and lbls is not None:
			ax.set_title('{}'.format(lbls[i][:7]))
		elif lbls is None and preds is not None:
			ax.set_title('{}'.format(preds[i][:7]))
		elif lbls is not None and preds is not None:
			ax.set_title('T: {} \nP: {}'.format(lbls[i][:7], preds[i][:7]))
	
	if extras != 0:
		for i in range(extras, cols):
			ax = axes[subplot_row, i]
			ax.remove()
	
	if preds is None and lbls is None:
		plt.subplots_adjust(wspace=0.1, hspace=-.4)
		
	if save_path is not None:
		plt.savefig(save_path, bbox_inches='tight')
		# _plt.gcf.get_active()
		plt.close(plt.gcf())
	else:
		plt.show(block=False)
		# add_color_bar(fig, ax)
		plt.pause(pause)


def choose_decoder(dataset):
	options = {
			   0: mnist_names, 'mnist': mnist_names, 'MNIST': mnist_names,
			   1: cifar10_names, "cifar10": cifar10_names, "CIFAR10": cifar10_names,
			   2: cifar100_names, "cifar100": cifar100_names, "CIFAR100": cifar100_names,
			   3: fmnist_names, "fashionmnist": fmnist_names, "FashionMNIST": fmnist_names,
			   "FMNIST": fmnist_names, "fmnist": fmnist_names,
			   4: gtsrb_names, "GTSRB": gtsrb_names, "gstrb": gtsrb_names,
			   5: imagenet_names, "ImageNet": imagenet_names, "imagenet": imagenet_names
			   }
	
	return options[dataset]

classMapPath = "DeepLearning/ImageDatasets/ImageNet/imagenet1000_clsidx_to_labels.txt"
f = open(classMapPath)
cls2idx = json.load(f)

keys = cls2idx.keys()
keys_i = []
for k in keys:
	keys_i.append( int(k) )

keys_s = sorted(keys_i)
i = 0
d = 0
for j in range(1000):
	if i != keys_s[j-d] :
		print(j)
		d = d+1
	i = i+1
g=0
def imagenet_names(labels, nmax=65):
	if type(labels) == torch.Tensor:
		labels = labels.detach().cpu().numpy()

	labels = np.ndarray.astype(labels, np.int)
	class_names = cls2idx
	c_names = np.array(np.copy(labels), dtype=object)
	for i in range(min(labels.shape[0], nmax)):
		c_names[i] = class_names[str(labels[i])]

	return c_names


def mnist_names(labels, nmax=65):
	if type(labels) == torch.Tensor:
		labels = labels.detach().cpu().numpy()

	labels = np.ndarray.astype(labels, np.int)
	class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
	c_names = np.array(np.copy(labels), dtype=object)
	for i in range(min(labels.shape[0], nmax)):
		c_names[i] = class_names[labels[i]]
	
	return c_names


def cifar10_names(labels, nmax=65):
	if type(labels) == torch.Tensor:
		labels = labels.detach().cpu().numpy()

	labels = np.ndarray.astype(labels, np.int)
	class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	c_names = np.array(np.copy(labels), dtype=object)
	for i in range(min(labels.shape[0], nmax)):
		c_names[i] = class_names[labels[i]]
	
	return c_names


def cifar100_names(labels, nmax=65):
	if type(labels) == torch.Tensor:
		labels = labels.detach().cpu().numpy()

	labels = np.ndarray.astype(labels, np.int)
	class_names = ['apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
				   'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
				   'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
				   'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
				   'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
				   'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion',
				   'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse',
				   'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear',
				   'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine',
				   'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
				   'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
				   'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table',
				   'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
				   'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman',
				   'worm'
				   ]
	c_names = np.array(np.copy(labels), dtype=object)
	for i in range(min(labels.shape[0], nmax)):
		c_names[i] = class_names[labels[i]]
	
	return c_names


def fmnist_names(labels, nmax=65):
	if type(labels) == torch.Tensor:
		labels = labels.detach().cpu().numpy()

	labels = np.ndarray.astype(labels, np.int)
	class_names = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
	c_names = np.array(np.copy(labels), dtype=object)
	for i in range(min(labels.shape[0], nmax)):
		c_names[i] = class_names[labels[i]]
	
	return c_names


def gtsrb_names(labels, nmax=65):
	if type(labels) == torch.Tensor:
		labels = labels.detach().cpu().numpy()

	labels = np.ndarray.astype(labels, np.int)
	class_names = ["SL (20km/h)", "SL (30km/h)", "SL (50km/h)", "SL (60km/h)", "SL (70km/h)", "SL (80km/h)",
				   "SL end (80km/h)", "SL (100km/h)", "SL (120km/h)", "No passing", "3.5 - No passing", "Right-of-way",
				   "Priority", "Yield", "Stop", "No vehicles", "3.5 - prohibited", "No entry",
				   "Caution", "Curve left", "Curve right", "Double curve", "Bumpy road", "Slippery road",
				   "Right narrows", "Road work", "Traffic signals", "Pedestrians", "Children crossing",
				   "Bicycle crossing",
				   "Ice/snow", "Animal crossing", "End speed and passing limits", "Right turn ahead", "Left turn ahead",
				   "Straight only",
				   "Straight or right", "Straight or left", "Keep right", "Keep left", "Roundabout mandatory",
				   "End no passing", "3.5 - End no passing"
				   ]
	c_names = np.array(np.copy(labels), dtype=object)
	for i in range(min(labels.shape[0], nmax)):
		c_names[i] = class_names[labels[i]]
	
	return c_names


def save_numpy_images(numpy_array, save_name, save_dir, ref_dir=""):
	path = os.path.join(ref_dir, save_dir)
	if not os.path.exists(path):
		os.makedirs(path)
	np.save(os.path.join(path, save_name), numpy_array)


def rssim(img1, img2, fh=3):
	(_, mean) = make_ssim(img1, img2, fh, True, False)
	return mean


def rssim_ind(img1, img2, fh=3):
	(img, mean) = make_ssim(img1, img2, fh, True, False)
	return mean


def psnr(img1, img2):
	mse = np.mean((img1 - img2)**2, (1, 2, 3))
	pixel_max = 1.0
	safe_div = np.divide(pixel_max, np.sqrt(mse), where=(np.sqrt(mse) != 0))
	psn = np.where(mse == 0, 100.0, 20*np.log10(safe_div, where=(safe_div != 0)))
	
	return np.mean(psn)


def rmse(img1, img2):
	mse = np.mean((img1 - img2)**2, (1, 2, 3))
	return np.mean(np.sqrt(mse))


class CutoutPIL(object):
	def __init__(self, cutout_factor=0.5):
		self.cutout_factor = cutout_factor

	def __call__(self, x):
		img_draw = ImageDraw.Draw(x)
		h, w = x.size[0], x.size[1]  # HWC
		h_cutout = int(self.cutout_factor * h + 0.5)
		w_cutout = int(self.cutout_factor * w + 0.5)
		y_c = np.random.randint(h)
		x_c = np.random.randint(w)

		y1 = np.clip(y_c - h_cutout // 2, 0, h)
		y2 = np.clip(y_c + h_cutout // 2, 0, h)
		x1 = np.clip(x_c - w_cutout // 2, 0, w)
		x2 = np.clip(x_c + w_cutout // 2, 0, w)
		fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

		return x

class PrefetchLoader:
	def __init__(self, loader):
		self.loader = loader
		self.stream = torch.cuda.Stream()

	def __iter__(self):
		first = True
		for batch in self.loader:
			with torch.cuda.stream(self.stream):  # stream - parallel
				self.next_input = batch[0].cuda(non_blocking=True) # note - (0-1) normalization in .ToTensor()
				self.next_target = batch[1].cuda(non_blocking=True)

			if not first:
				yield input, target  # prev
			else:
				first = False

			torch.cuda.current_stream().wait_stream(self.stream)
			input = self.next_input
			target = self.next_target

			# Ensures that the tensor memory is not reused for another tensor until all current work queued on stream are complete.
			input.record_stream(torch.cuda.current_stream())
			target.record_stream(torch.cuda.current_stream())

		# final batch
		yield input, target

		# cleaning at the end of the epoch
		del self.next_input
		del self.next_target
		self.next_input = None
		self.next_target = None

	def __len__(self):
		return len(self.loader)

	@property
	def sampler(self):
		return self.loader.sampler

	@property
	def dataset(self):
		return self.loader.dataset

	def set_epoch(self, epoch):
		self.loader.sampler.set_epoch(epoch)



if __name__ == '__main__':
	print("Testing ImageFns Plot")
	import Datasets as Ds
	ds = Ds.datasetHandler('C:\\Users\\jfcle\\PycharmProjects\\SharedFolder\\Datasets\\')
	ds.load_image_dataset("cifar10")
	
	plot_grid(ds.datalib["tset"][0], imgsize=ds.datalib['imgsize'])

g = 0
