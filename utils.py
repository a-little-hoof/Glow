import torch
import numpy as np
import os
import glob
import imageio as io
import json
import sys
import ast

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()
        return self.terminal.flush()
        
def setup_saver(results_dir, identifier):
    folder_no = len(glob.glob(f'{results_dir}/*'))
    folder = f'{results_dir}/{folder_no:03}-{identifier}'
    os.makedirs(folder, exist_ok=True)
    return folder

def save_images(images, path, imrange=[0,1]):

    images = np.squeeze(images)
    if len(images.shape) == 4:
        images = np.swapaxes(images, 1, 2)
        images = np.swapaxes(images, 2, 3)
        if images.shape[-1] == 2:
            images = images[...,0] + 1.j*images[...,1]
    # images = images.detach().numpy()
    N = int(np.sqrt(len(images)))
    imrows = [ np.concatenate(images[N*i:N*i+N], axis=1) for i in range(N) ]
    im = np.concatenate(imrows)
    im = np.clip(im, *imrange)
    im = (im - imrange[0]) / (imrange[1] - imrange[0]) * 255
    im = abs(im)
    im = im.astype(np.uint8)
    io.imsave(path, im)
    return im
