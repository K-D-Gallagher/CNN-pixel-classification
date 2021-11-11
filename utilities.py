import torch
from torch.autograd import Function

device_f = torch.device("cuda:0")


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = (
                grad_output
                * 2
                * (target * self.union - self.inter)
                / (self.union * self.union)
            )
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device_f).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


import torch.nn.functional as F
from tqdm import tqdm

classes = 1


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val



################################################################################
# class for preprocessing images
################################################################################

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import logging
from PIL import Image

class PreProcessData(Dataset):

    # constructor - this is where we take the arguments
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=""):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        # splits filename by extension and only holds onto the name
        self.ids = [
            splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith(".")
        ]
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    # this is called overloading:
    # redefining the length function for this class as the number of files, rather
    # than the length of the filepath, which it would be otherwise
    def __len__(self):
        return len(self.ids)

    ##############
    #
    ##############

    # class method means that this function is only available within this class
    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"  # default 0.5
        pil_img = pil_img.resize((newW, newH))  # rescaled image

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    ###########
    #
    ############
    def __getitem__(self, i):  # matches mask and image and preprosses images

        # within this class, how do we pull out the element at this index
        idx = self.ids[i]

        mask_file = glob(self.masks_dir + idx + "_mask.*")
        img_file = glob(self.imgs_dir + idx + ".*")

        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {idx}: {img_file}"
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert (
            img.size == mask.size
        ), f"Image and mask {idx} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            "image": torch.from_numpy(img).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask).type(torch.FloatTensor),
        }  # Preprocess the image and can open image
