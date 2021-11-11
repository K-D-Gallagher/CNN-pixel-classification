import argparse
import logging
import os
from os.path import splitext
from os import listdir
from glob import glob

from utilities import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp  # Use smp.Model in place of UNet

# Plots image and prediction (mask) side by side
def plot_img_and_mask(img, mask):
    classes_n = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes_n + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    if classes_n > 1:
        for i in range(classes_n):
            ax[i + 1].set_title(f"Output mask (class {i+1})")
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f"Output mask")
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def predict_img(net, classes, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()  # Disables dropout

    img = torch.from_numpy(
        PreProcessData.preprocess(full_img, scale_factor)
    )  # Creates torch tensor from np array

    img = img.unsqueeze(0)
    img = img.to(
        device=device, dtype=torch.float32
    )  # Moves the image to the GPU (or CPU is selected)

    with torch.no_grad():
        output = net(img)

        if (
            classes > 1
        ):  # Sets the propper activation function based on the number of classes
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor(),
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    if args.mask_threshold:  # Choose to binarize image or get raw prediction
        return full_mask > out_threshold
    else:
        return full_mask


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks from input images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="MODEL.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        nargs="+",
        help="filenames of input images",
        required=True,
    )

    parser.add_argument(
        "--output", "-o", metavar="INPUT", nargs="+", help="Filenames of ouput images"
    )
    parser.add_argument(
        "--viz",
        "-v",
        action="store_true",
        help="Visualize the images as they are processed",
        default=False,
    )
    parser.add_argument(
        "--no-save",
        "-n",
        action="store_true",
        help="Do not save the output masks",
        default=False,
    )
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        help="Minimum probability value to consider a mask pixel white",
        default=None,
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        help="Scale factor for the input images",
        default=0.5,
    )
    parser.add_argument(
        "--classes", "-c", type=int, help="Model output channels", default=1
    )
    parser.add_argument(
        "--in-channels", "-ic", type=int, help="Model input channels", default=1
    )
    parser.add_argument(
        "--device", "-d", type=str, help="Select device", default="cuda:0"
    )
    parser.add_argument(
        "--encoder", "-en", type=str, help="Name of encoder", default="resnet34"
    )
    parser.add_argument(
        "--weight", "-wt", type=str, help="Encoder weights", default=None
    )
    parser.add_argument("--architecture", "-a", type=str, help="Name of architecture")

    return parser.parse_args()


# Fucntion for naming prediction image
def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


# Converts array to image
def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))
    # return Image.fromarray((255.0 / mask.max() * (mask - mask.min())).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_file = args.input

    in_files_str = in_file[0]

    in_file_lett = list(in_files_str)
    file_type = ("".join(in_file_lett[-4:])).lower()

    if file_type == ".tif":
        in_files = in_file
    else:
        in_files_nm = sorted(os.listdir(in_files_str))
        in_files = []
        for file in in_files_nm:
            name = in_files_str + "/" + file
            in_files.append(name)

    # out_files = get_output_filenames(args)

    classes = args.classes
    in_channels = args.in_channels
    encoder = args.encoder
    weight = args.weight
    architecture = args.architecture

    # Architecture must be the same as the architecture used to train the model
    # net = smp.Unet(encoder_name="resnet18", in_channels=in_channels, classes=classes, encoder_weights="imagenet")

    def arch_arg(architecture):
        if architecture.lower() == "unet":
            net = smp.Unet(
                encoder_name=encoder,
                in_channels=in_channels,
                classes=classes,
                encoder_weights=weight,
            )

        elif architecture.lower() == "unetplusplus":
            net = smp.UnetPlusPlus(
                encoder_name=encoder,
                in_channels=in_channels,
                classes=classes,
                encoder_weights=weight,
            )

        elif architecture.lower() == "manet":
            net = smp.MAnet(
                encoder_name=encoder,
                in_channels=in_channels,
                classes=classes,
                encoder_weights=weight,
            )

        elif architecture.lower() == "linknet":
            net = smp.Linknet(
                encoder_name=encoder,
                in_channels=in_channels,
                classes=classes,
                encoder_weights=weight,
            )

        elif architecture.lower() == "fpn":
            net = smp.FPN(
                encoder_name=encoder,
                in_channels=in_channels,
                classes=classes,
                encoder_weights=weight,
            )

        else:
            print("Architecture not recognized.")
            quit()

        return net

    net = arch_arg(architecture)

    logging.info("Loading model {}".format(args.model))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):  # Ability to predict multiple images
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(
            net=net,
            classes=classes,
            full_img=img,
            scale_factor=args.scale,
            out_threshold=args.mask_threshold,
            device=device,
        )

        if not args.no_save:

            # TODO: Update the naming function so it is not here
            fn_split = fn.split(".")
            if not args.output:
                out_fn = fn_split[0] + "_OUT.tif"
            else:
                out_fn_ls = args.output
                out_fn = out_fn_ls[0]

            result = mask_to_image(mask)

            result.save(out_fn)

            logging.info("Mask saved to {}".format(out_fn))

        if args.viz:
            logging.info(
                "Visualizing results for image {}, close to continue ...".format(fn)
            )
            plot_img_and_mask(img, mask)
