import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utilities import *
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

import datetime







dir_checkpoint = None


################################################################################
################################################################################
#
#
# function to train model
#
#
################################################################################
################################################################################
def train_net(
    net,
    device,
    epochs=5,
    batch_size=1,
    lr=0.001,
    val_percent=0.1,
    save_cp=True,
    img_scale=0.5,
):

    ############################################################################
    # preprocess data and split into training and validation sets
    ############################################################################

    # preprocess dataset
    dataset = PreProcessData(dir_img, dir_mask, img_scale)

    # specify number of images for training and validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    # split into training and validation sets
    train, val = random_split(dataset, [n_train, n_val])  # Train/validation split

    # Pytorch data loader. Try setting num_workers to run two scripts in parallel
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    ############################################################################
    # tensorboard visualization
    ############################################################################

    #
    writer = SummaryWriter(
        log_dir=dir_checkpoint, comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}"
    )
    global_step = 0

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    """
    )


    ############################################################################
    # define optimizer, scheduler, and loss function
    ############################################################################

    # Essentially gradient decent with momentum (adaptive learning rate)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # Dynamic learning rate based on validation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min" if classes > 1 else "max", patience=10
    )

    # define loss according to number of classes
    if classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()



    ############################################################################
    # training loop
    ############################################################################
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        ########################################################################
        # tqdm package for showing training progress metrics in the command line
        ########################################################################
        # with / except: https://realpython.com/python-with-statement/
        # runtime manager: makes sure that when you open a file with a function,
        # this makes sure you close it when you're done
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:  # Shows progress of scepific functions in trining loop



            ####################################################################
            # loop through Pytorch dataloader
            ####################################################################
            for batch in train_loader:



                ################################################################
                # load current batch of images and move to GPU
                ################################################################

                # pull out current batch of images and corresponding ground truth masks
                imgs = batch["image"]
                true_masks = batch["mask"]

                # test to make sure the loaded image has the correct number of
                # channels and trip and error if this condition is not met
                assert imgs.shape[1] == in_channels, (
                    f"Network has been defined with {in_channels} input channels, "
                    f"but loaded images have {imgs.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                # move images to specified device (GPU)
                imgs = imgs.to(
                    device=device, dtype=torch.float32
                )

                # define mask datatype and move to specified device
                mask_type = torch.float32 if classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)



                ################################################################
                # pass to model for prediction and calculate loss
                ################################################################

                # Pass the images through the specified CNN
                masks_pred = net(imgs)

                # calculate loss (criterion is defined above)
                loss = criterion(
                    masks_pred, true_masks
                )

                # update total epoch loss
                epoch_loss += loss.item()



                ################################################################
                # log & print loss via tqdm package and TensorBoard
                ################################################################

                # Adding loss values to log of TensorBoard
                writer.add_scalar(
                    "Loss/train", loss.item(), global_step
                )

                # print loss to terminal via tqdm package
                pbar.set_postfix(**{"loss (batch)": loss.item()})



                ################################################################
                #
                ################################################################

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()



                ################################################################
                # update tqdm package and TensorBoard every XXXX iteration
                ################################################################

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace(".", "/")
                        writer.add_histogram(
                            "weights/" + tag, value.data.cpu().numpy(), global_step
                        )
                        writer.add_histogram(
                            "grads/" + tag, value.grad.data.cpu().numpy(), global_step
                        )
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar(
                        "learning_rate", optimizer.param_groups[0]["lr"], global_step
                    )

                    if classes > 1:
                        logging.info("Validation cross entropy: {}".format(val_score))
                        writer.add_scalar("Loss/test", val_score, global_step)
                    else:
                        logging.info("Validation Dice Coeff: {}".format(val_score))
                        writer.add_scalar("Dice/test", val_score, global_step)

                    writer.add_images("images", imgs, global_step)
                    if classes == 1:
                        writer.add_images("masks/true", true_masks, global_step)
                        writer.add_images(
                            "masks/pred", torch.sigmoid(masks_pred) > 0.5, global_step
                        )


        ########################################################################
        #
        ########################################################################

        # make sure save_cp == True
        if save_cp:
            # Saving model every 5 epochs
            if epoch % 5 == 0:
                # make directory if it doesn't exist
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                # save checkpoint
                torch.save(
                    net.state_dict(), dir_checkpoint + f"CP_epoch{epoch + 1}.pth"
                )
                # print confirmation
                logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()


####################################################################
# command line interface tool - for asking for help from commandline
####################################################################
def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=5,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=1,
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "-s",
        "--scale",
        dest="scale",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "-v",
        "--validation",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "-c", "--classes", type=int, help="Model output channels", default=1
    )
    parser.add_argument(
        "-ic", "--in-channels", type=int, help="Model input channels", default=1
    )
    parser.add_argument(
        "-d", "--device", type=str, help="Select device", default="cuda:0"
    )
    parser.add_argument(
        "-cp",
        "--checkpoint",
        type=str,
        help="Name folder for checkpoints",
        default="checkpoints/",
    )
    parser.add_argument(
        "-fn", "--file", type=str, help="Name folder for images", default=None
    )
    parser.add_argument(
        "-mf", "--mask-folder", type=str, help="Name for folder for mask", default=None
    )
    parser.add_argument(
        "-en", "--encoder", type=str, help="Name of encoder", default="resnet34"
    )
    parser.add_argument(
        "-wt", "--weight", type=str, help="Encoder weights", default=None
    )
    parser.add_argument("-a", "--architecture", type=str, help="Name of architecture")

    return parser.parse_args()



################################################################################
################################################################################
################################################################################
#
#
# actual start of script - i.e. actually running the model
# everything prior to this is just definitions
#
#
################################################################################
################################################################################
################################################################################


# checks if this script is being run or not
# if its just being imported, only import functions - don't actually run script
if __name__ == "__main__":

    # Initiate logging of metrics
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    # Function to call arguments from argparse
    args = get_args()

    # Define the device
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    # device = torch.device('cuda:1') # add argparser for this

    logging.info(f"Using device {device}")
    dir_checkpoint = args.checkpoint
    dir_img = args.file
    dir_mask = args.mask_folder

    # Change here to adapt to your data
    # in_channels=3 for RGB images
    # classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use classes=1
    #   - For 2 classes, use classes=1
    #   - For N > 2 classes, use classes=N
    in_channels = args.in_channels
    classes = args.classes
    encoder = args.encoder
    weight = args.weight
    architecture = args.architecture
    # net = smp.Unet(encoder_name=encoder, in_channels=in_channels, classes=classes, encoder_weights=weight)


    ###########################################################################################
    # command line arguments for specifying which architectures to use from segmentation models
    ###########################################################################################
    # alt to using arg_paser is yaml files

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

    logging.info(
        f"Network:\n"
        f"\t{in_channels} input channels\n"
        f"\t{classes} output channels (classes)\n"
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    #############################################
    # saves model if keyboard interruption occurs
    #############################################
    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
