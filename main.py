# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from Model import MyNet
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import glob

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()


def decode_segmap(image, nc=21):

    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def crop(image, source, nc=21):
    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)

    foreground = source

    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))

    background = 255 * np.ones_like(rgb).astype(np.uint8)

    foreground = foreground.astype(float)
    background = background.astype(float)

    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

    alpha = alpha.astype(float) / 255

    foreground = cv2.multiply(alpha, foreground)

    background = cv2.multiply(1.0 - alpha, background)

    outImage = cv2.add(foreground, background)

    return outImage / 255


def modify(source):

    """Use the below commented code to upload an image using colab.upload in colab notebook but then
    there is no need to use img parameter and source parameter while using crop function"""
    img = Image.open(source)
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    trf = T.Compose(
        [
            T.Resize(640),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    inp = trf(img).unsqueeze(0)
    out = dlab(inp)["out"]
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    result = decode_segmap(om)
    # plt.imshow(result)
    # plt.axis("off")
    # plt.show()

    path, name = source.split("/")
    path += "/automatiquely_segmented"

    Path(path).mkdir(parents=True, exist_ok=True)

    plt.imsave(f"{path}/{name}", result)


use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description="PyTorch Unsupervised Segmentation")

parser.add_argument(
    "--scribble", action="store_true", default=False, help="use scribbles"
)
parser.add_argument(
    "--nChannel", metavar="N", default=100, type=int, help="number of channels"
)
parser.add_argument(
    "--maxIter",
    metavar="T",
    default=1000,
    type=int,
    help="number of maximum iterations",
)
parser.add_argument(
    "--minLabels", metavar="minL", default=3, type=int, help="minimum number of labels"
)
parser.add_argument("--lr", metavar="LR", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--nConv", metavar="M", default=2, type=int, help="number of convolutional layers"
)
parser.add_argument(
    "--visualize", metavar="1 or 0", default=1, type=int, help="visualization flag"
)
parser.add_argument(
    "--input", metavar="FILENAME", help="input image file name", required=True
)
parser.add_argument(
    "--stepsize_sim",
    metavar="SIM",
    default=1,
    type=float,
    help="step size for similarity loss",
    required=False,
)
parser.add_argument(
    "--stepsize_con",
    metavar="CON",
    default=1,
    type=float,
    help="step size for continuity loss",
)
parser.add_argument(
    "--stepsize_scr",
    metavar="SCR",
    default=0.5,
    type=float,
    help="step size for scribble loss",
)
args = parser.parse_args()


paths = glob.glob(f"{args.input}/*.jpg")
# for path in paths:
#     modify(path)


# load image
for path in paths:
    im = cv2.imread(path)
    im = cv2.resize(
        im, (im.shape[1] // 4, im.shape[0] // 4), interpolation=cv2.INTER_AREA
    )
    data = torch.from_numpy(
        np.array([im.transpose((2, 0, 1)).astype("float32") / 255.0])
    )
    data = data.cuda() if use_cuda else data.cpu()

    # load scribble
    if args.scribble:
        mask = cv2.imread(
            args.input.replace("." + args.input.split(".")[-1], "_scribble.png"), -1
        )
        mask = mask.reshape(-1)
        mask_inds = np.unique(mask)
        mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
        inds_sim = torch.from_numpy(np.where(mask == 255)[0])
        inds_scr = torch.from_numpy(np.where(mask != 255)[0])
        target_scr = torch.from_numpy(mask.astype(np.int_))

        if use_cuda:
            inds_sim = inds_sim.cuda()
            inds_scr = inds_scr.cuda()
            target_scr = target_scr.cuda()

        # set minLabels
        args.minLabels = len(mask_inds)

    # train
    model = MyNet(data.size(1), args.nChannel, args.nConv)
    model.cuda() if use_cuda else model.cpu()
    model.train()

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # scribble loss definition
    loss_fn_scr = torch.nn.CrossEntropyLoss()

    # continuity loss definition

    loss_hpy = torch.nn.L1Loss(reduction="mean")
    loss_hpz = torch.nn.L1Loss(reduction="mean")

    HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], args.nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, args.nChannel)

    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    label_colours = np.random.randint(255, size=(100, 3))

    for batch_idx in range(args.maxIter):
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

        outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)

        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if args.visualize:
            im_target_rgb = np.array(
                [label_colours[c % args.nChannel] for c in im_target]
            )
            im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
            cv2.imshow("output", im_target_rgb)
            cv2.waitKey(10)

        # loss
        if args.scribble:
            loss = (
                args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim])
                + args.stepsize_scr
                * loss_fn_scr(output[inds_scr], target_scr[inds_scr])
                + args.stepsize_con * (lhpy + lhpz)
            )
        else:
            loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (
                lhpy + lhpz
            )

        loss.backward()
        optimizer.step()

        print(
            f"{batch_idx} / {args.maxIter} | label num : {nLabels} | loss : {loss.item()}",
        )

        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break

    # save output image
    if not args.visualize:
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

    path, name = path.split("/")
    path += "manually_segmented"
    Path(path).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{path}/{name}", im_target_rgb)
