import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
import json

class Coco_paths():
    def __init__(self,model_path):
        self.base_path = './ImageDatasets/COCO/val2017/'
        self.IDs_path = './ImageDatasets/COCO/annotations/'
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
        self.all_images = os.listdir( self.base_path )
        self.current_image = 0
        self.total_image = len(self.all_images)

    def open_image(self,N):
        img_path = self.base_path + self.all_images[N]

        image = Image.open(img_path)
        inputs = self.feature_extractor (images=image, return_tensors="pt")

        img = inputs['pixel_values'].cuda()
        img.requires_grad = True

        return img

    def open_image_by_filename(self,filename):
        img_path = self.base_path + filename

        image = Image.open(img_path)
        input = self.feature_extractor (images=image, return_tensors="pt")

        return input['pixel_values']

    def open_N_image_by_filename(self,N,json_datafile=None):
        if json_datafile is None:
            datafile_path = self.IDs_path + 'stuff_val2017.json'
        else:
            datafile_path = self.IDs_path + json_datafile

        datafile = open(datafile_path)
        data = json.load(datafile)

        images = None
        for i in range(N):
            # label = data['images'][i]['file_name']
            filename = data['images'][i]['file_name']
            image = self.open_image_by_filename(filename)

            if images is not None:
                images = torch.cat( [images,image],0 )
            else:
                images = image

        return images



    def get_n_images(self,N):
        images = torch.cat([self.open_image(self.current_image + i) for i in range(N)])
        self.current_image = self.current_image + N

        return images

    def restart(self,start=None):
        if start is None:
            self.current_image = 0
        else:
            self.current_image = start





def open_image(img_path):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    image = Image.open(img_path)
    inputs = feature_extractor(images=image, return_tensors="pt")

    img = inputs['pixel_values'].cuda()
    img.requires_grad = True

    return img


def open_VIT(name):
    model = ViTForImageClassification.from_pretrained(name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    # model.train()

    return model

def open_optimizer(model):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    optimizer.zero_grad()

    return optimizer, loss_fn


def torch_to_image(t_tensor):
    reduced = np.swapaxes(np.swapaxes(t_tensor.detach().cpu(),1,3),1,2)
    image = (reduced-reduced.min())/(reduced.max()-reduced.min())

    return image

def plot_grid(imgs, lbls=None, preds=None, cols=13, img_size=(224, 224, 3), nmax=65, save_path=None):
    """
    This function plots images in google colab.

    Inputs:
        imgs - A numpy ndarray of the images to be plotted
      lbls(optional) - ground truth labels of imgs
      preds(optional) - predicted labes of imgs
      cols(default: 13) - the number of images to be plotted on each row
      img_sie(default: (28,28,1)) - the size of each element in imgs
      nmax(default: 65) - a limit on the number of imgs to print (the default prints 5 rows of 13)

    """
    if type(imgs) == type(torch.tensor(1)):
        imgs = torch_to_image(imgs)
    figu = plt.figure()
    img_count = min(imgs.shape[0], nmax)
    cols = min(cols,img_count)
    rows = int(-(-img_count // cols))
    extras = img_count % cols


    img_width = min(2 * cols, 26)
    img_height = img_width / cols * 1.25 * rows

    # _plt.clf()
    fig, axes = plt.subplots(rows, cols,
                              figsize=(img_width, img_height),
                              sharex=True, sharey=True, squeeze=False,
                              subplot_kw=dict(aspect='equal'))

    for i in range(img_count):
        subplot_row = i // cols
        subplot_col = i % cols

        ax = axes[subplot_row, subplot_col]

        if img_size[2] == 1:
            cm = 'Greys_r'
            plottable_image = np.reshape(imgs[i], img_size[0:2])
        else:
            cm = None
            plottable_image = np.reshape(imgs[i], img_size)

        fig = ax.imshow(plottable_image, vmin=0.0, vmax=1.0, cmap='Greys_r')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        if preds is None and lbls is not None:
            ax.set_title('True: {}'.format(lbls[i]))
        elif lbls is None and preds is not None:
            ax.set_title('Prediction: {}'.format(preds[i]))
        elif lbls is not None and preds is not None:
            ax.set_title('True: {} \nPrediction: {}'.format(lbls[i], preds[i]))

    if extras != 0:
        for i in range(extras, cols):
            ax = axes[subplot_row, i]
            ax.remove()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
