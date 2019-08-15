"""
UNet
Common utility functions and classes
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from skimage import io
from PIL import Image
import numpy as np
import torch
from skimage.morphology import label
import pandas as pd
import matplotlib.pylab as plt
import glob


# Base Configuration class
# Don't use this class directly. Instead, sub-class it and override

class Config():

    name = None

    img_width = 256
    img_height = 256

    img_channel = 3

    batch_size = 16

    learning_rate = 1e-3
    learning_momentum = 0.9
    weight_decay = 1e-4

    shuffle = False

    def __init__(self):
        self.IMAGE_SHAPE = np.array([
            self.img_width, self.img_height, self.img_channel
        ])

    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

# Configurations

class Option(Config):
    """Configuration for training on Kaggle Data Science Bowl 2018
    Derived from the base Config class and overrides specific values
    """
    name = "DSB2018"

    # root dir of training and validation set
    root_dir = '/Users/mahdi/software/Colon_Mask_RCNN/mrcnn/dataset/new_crypts/train/combined'

    # root dir of testing set
    test_dir = '/Users/mahdi/software/Colon_Mask_RCNN/mrcnn/dataset/new_crypts/valid/testing_data'

    # save segmenting results (prediction masks) to this folder
    results_dir = '/Users/mahdi/software/Colon_Mask_RCNN/mrcnn/dataset/new_crypts/results'

    num_workers = 1     	# number of threads for data loading
    shuffle = True      	# shuffle the data set
    batch_size = 16     		# GTX1060 3G Memory
    epochs = 2			# number of epochs to train
    is_train = True     	# True for training, False for making prediction
    save_model = True   	# True for saving the model, False for not saving the model

    n_gpu = 1				# number of GPUs

    learning_rate = 1e-3	# learning rage
    weight_decay = 1e-4		# weight decay

    pin_memory = True   	# use pinned (page-locked) memory. when using CUDA, set to True

    is_cuda = torch.cuda.is_available()  	# True --> GPU
    num_gpus = torch.cuda.device_count()  	# number of GPUs
    checkpoint_dir = "./checkpoint"  		# dir to save checkpoints
    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type

"""
Dataset orgnization:
Read images and masks, combine separated mask into one
Write images and combined masks into specific folder
"""
class Utils(object):
    """
    Initialize image parameters from DSB2018Config class
    """
    def __init__(self, stage1_train_src, stage1_train_dest, stage1_test_src, stage1_test_dest):
        self.opt = Option
        self.stage1_train_src = stage1_train_src
        self.stage1_train_dest = stage1_train_dest
        self.stage1_test_src = stage1_test_src
        self.stage1_test_dest = stage1_test_dest

    # Combine all separated masks into one mask
    def assemble_masks(self, path, id):
        # mask = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH), dtype=np.uint8)
        mask = None
        for i, mask_file in enumerate(glob.glob(path + id + '_*')):
            # mask_ = Image.open(mask_file).convert("RGB")
            mask_ = Image.open(mask_file)
            # mask_ = mask_.resize((self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
            mask_ = np.asarray(mask_)
            if i == 0:
                mask = mask_
                continue
            mask = mask | mask_
        # mask = np.expand_dims(mask, axis=-1)
        return mask

    # read all training data and save them to other folder
    def prepare_training_data(self):
        # get imageId
        train_ids = next(os.walk(self.stage1_train_src + '/Images'))[2]

        # read training data
        X_train = []
        Y_train = []
        print('reading training data starts...')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids)):
            id = os.path.splitext(id_)[0]
            dest = os.path.join(self.stage1_train_dest, id)
            if not os.path.exists(dest):
                os.mkdir(dest)
            img = Image.open(os.path.join(self.stage1_train_src, 'Images', id_)).convert("RGB")
            mask = self.assemble_masks(self.stage1_train_src + '/Annotation/', id)
            img.save(os.path.join(dest, 'image.png'))
            Image.fromarray(mask).save(os.path.join(dest, 'mask.png'))

        print('reading training data done...')

    # read testing data and save them to other folder
    def prepare_testing_data(self):
        # get imageId
        test_ids = next(os.walk(self.stage1_test_src + '/Images'))[2]
        # read training data
        print('reading testing data starts...')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(test_ids)):
            id = os.path.splitext(id_)[0]
            dest = os.path.join(self.stage1_test_dest, id)
            if not os.path.exists(dest):
                os.mkdir(dest)
            img = Image.open(os.path.join(self.stage1_test_src, 'Images', id_)).convert("RGB")
            img.save(os.path.join(dest, 'image.png'))

        print('reading testing data done...')


def compute_iou(predictions, img_ids, val_loader):
    """
    compute IOU between two combined masks, this does not follow kaggle's evaluation
    :return: IOU, between 0 and 1
    """
    ious = []
    for i in range(0, len(img_ids)):
        pred = predictions[i]
        img_id = img_ids[i]
        mask_path = os.path.join(Option.root_dir, img_id, 'mask.png')
        mask = np.asarray(Image.open(mask_path).convert('L'), dtype=np.bool)
        union = np.sum(np.logical_or(mask, pred))
        intersection = np.sum(np.logical_and(mask, pred))
        iou = intersection/union
        ious.append(iou)
    df = pd.DataFrame({'img_id':img_ids,'iou':ious})
    df.to_csv('IOU.csv', index=False)



# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def encode_and_save(preds_test_upsampled, test_ids):
    """
    Use run-length-encoding encode the prediction masks and save to csv file for submitting
    :param preds_test_upsampled: list, for each elements, numpy array (Width, Height)
    :param test_ids: list, for each elements, image id
    :return:
        save to csv file
    """
    # save as imgs
    for i in range(0, len(test_ids)):
        path = os.path.join(Option.results_dir, test_ids[i])
        if not os.path.exists(path):
            os.mkdir(path)
        # Image.fromarray(preds_test_upsampled[i]).save(os.path.join(path,'prediction.png'))
        plt.imsave(os.path.join(path, 'prediction.png'),preds_test_upsampled[i], cmap='gray')
    # save as encoding
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub-dsbowl2018.csv', index=False)

if __name__ == '__main__':
    """ Prepare training data and testing data
    read data and overlay masks and save to destination path
    """
    stage1_train_src = '/Users/mahdi/software/Colon_Mask_RCNN/mrcnn/dataset/new_crypts/train'
    stage1_train_dest = '/Users/mahdi/software/Colon_Mask_RCNN/mrcnn/dataset/new_crypts/train/combined'
    stage1_test_src = '/Users/mahdi/software/Colon_Mask_RCNN/mrcnn/dataset/new_crypts/valid'
    stage1_test_dest = '/Users/mahdi/software/Colon_Mask_RCNN/mrcnn/dataset/new_crypts/valid/testing_data'

    util = Utils(stage1_train_src, stage1_train_dest, stage1_test_src, stage1_test_dest)
    util.prepare_training_data()
    util.prepare_testing_data()