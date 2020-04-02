from matplotlib import pyplot as plt
import os
import numpy as np
import imageio
# import imgaug as ia
import imgaug.augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.batches import UnnormalizedBatch
import random
import time
from skimage.util.shape import view_as_windows

inDir = '/content/drive/My Drive/Capstone/Open Cities Competition/Training Folder/datasets/Meta-Chips'


def find_datasets(root):
    table = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            table.append(name)
    n_sub_sets = len(table)
    n_dim = int(n_sub_sets / 2)
    table = sorted(table)
    # print(table)
    table = np.array(table).reshape(2, n_dim)
    return table


def window_creation(chips_x, chips_y, ISZ):
    dim_Max = chips_x.shape[0]
    windows = dim_Max // ISZ
    x_windows = view_as_windows(chips_x, (ISZ, ISZ, 3), ISZ)[:windows, :windows, 0, :, :]
    y_windows = view_as_windows(chips_y, (ISZ, ISZ), ISZ)[:windows, :windows, :, :]
    return x_windows, y_windows, windows


def chip_verification(x_windows, y_windows, dim_Max, ISZ):
    x_chips_verified = []
    y_chips_verified = []
    for x in range(dim_Max):
        for y in range(dim_Max):
            label_coverage = np.count_nonzero(y_windows[x][y]) / (ISZ * ISZ)
            if label_coverage > .05 and label_coverage < .95:
                x_chips_verified.append(x_windows[x][y])
                y_chips_verified.append(y_windows[x][y])
    return np.array(x_chips_verified), np.expand_dims(np.array(y_chips_verified), axis=3)


def pad_valid(x_chips_verified, y_chips_verified, amt):
    while len(x_chips_verified) < amt:
        x_chips_verified = np.append(x_chips_verified, x_chips_verified, axis=0)
        y_chips_verified = np.append(y_chips_verified, y_chips_verified, axis=0)
    return x_chips_verified, y_chips_verified


def stretch_n(bands, lower_percent=7, higher_percent=93):
    out = np.zeros_like(bands).astype(np.uint8)  # added .astype(np.float32) per comment within thread
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / ((d - c) + 1e-7)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    # print('stretched!')
    return out.astype(np.uint8)


def simple_seq():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])
    return seq


def aug_seq():
    seq = iaa.Sequential([
        iaa.Fliplr(0.4),  # horizontal flips
        iaa.Flipud(0.4),  # vertical flips
        # iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        # iaa.Sometimes(
        #    0.3,
        #    iaa.GaussianBlur(sigma=(0, 0.5))
        # ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # iaa.Sometimes(
        #    0.5,
        #    iaa.Multiply((0.8, 1.2), per_channel=0.2)
        # ),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Sometimes(
            0.5,
            iaa.Affine(
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8))  # to add shear back, remove the second ) above, and add the , back as well
        ),
        iaa.Sometimes(
            0.10,
            iaa.OneOf([
                iaa.Sequential([
                    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                    iaa.WithChannels(0, iaa.Add((0, 50))),
                    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                iaa.Sequential([
                    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                    iaa.WithChannels(1, iaa.Add((0, 50))),
                    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                iaa.Sequential([
                    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                    iaa.WithChannels(2, iaa.Add((0, 50))),
                    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                iaa.WithChannels(0, iaa.Add((0, 50))),
                iaa.WithChannels(1, iaa.Add((0, 50))),
                iaa.WithChannels(2, iaa.Add((0, 50)))
            ])
        )

    ], random_order=True)  # apply augmenters in random order
    return seq


def batch_op_sub_chip_augmentation(x_chips_total, y_chips_total, amt=1000, test=True, test_size=256, ISZ=256,
                                   random_seed=42):
    if test:
        time_start = time.time()
        amt = test_size
        amt = int(amt // 128 * 128)
        print('test selected, amt:', amt)
    else:
        amt = int(amt // 128 * 128)
        print(amt, 'non-test amt here')

    dataset_table = find_datasets(inDir)

    if len(x_chips_total) < 50:
        for index in range(len(dataset_table[0])):
            selected_x = dataset_table[0][index]
            selected_y = dataset_table[1][index]
            # seed = 15
            # seed = np.random.randint(1,5000)
            print(selected_x, selected_y)

            chips_x = np.load(inDir + '/x_set/' + selected_x)
            chips_y = np.load(inDir + '/y_set/' + selected_y)

            x_windows, y_windows, dim_Max = window_creation(chips_x, chips_y, ISZ)
            del chips_x, chips_y
            # print(dim_Max)
            x_chips_verified, y_chips_verified = chip_verification(x_windows, y_windows, dim_Max, ISZ)
            del x_windows, y_windows
            # if test:
            #  print(x_chips_verified.shape, y_chips_verified.shape)

            #        np.random.seed(42)
            #        np.random.shuffle(dataset_table[0])
            #        np.random.seed(42)
            #        np.random.shuffle(dataset_table[1])
            np.random.seed(random_seed)
            np.random.shuffle(x_chips_verified)
            np.random.seed(random_seed)
            np.random.shuffle(y_chips_verified)
            print(x_chips_verified.shape, y_chips_verified.shape)
            if len(x_chips_verified) < 500:
                x_chips_verified, y_chips_verified = pad_valid(x_chips_verified, y_chips_verified, 100)
            # x_chips_verified = x_chips_verified[:500]
            # y_chips_verified = y_chips_verified[:500]

            if index == 0:
                x_chips_total = x_chips_verified
                y_chips_total = y_chips_verified

            if index > 0:
                x_chips_total = np.append(x_chips_total, x_chips_verified, axis=0)
                y_chips_total = np.append(y_chips_total, y_chips_verified, axis=0)
                del x_chips_verified, y_chips_verified
            print(x_chips_total.shape, y_chips_total.shape)
            if test:
                break

    np.random.seed(random_seed * 2)
    np.random.shuffle(x_chips_total)
    np.random.seed(random_seed * 2)
    np.random.shuffle(y_chips_total)
    print(x_chips_total.shape, y_chips_total.shape)

    # x_chips_padded = x_chips_total#[:amt]
    # y_chips_padded = y_chips_total#[:amt]
    # del x_chips_total, y_chips_total
    # print(x_chips_padded.shape, y_chips_padded.shape)

    BATCH_SIZE = 128
    NB_BATCHES = int(amt / 128)

    images_batch = [x_chips_total[_] for _ in range(BATCH_SIZE)]
    segmentation_maps_batch = [y_chips_total[_] for _ in range(BATCH_SIZE)]

    # images_batch = [x_chips_padded[_] for _ in range(BATCH_SIZE)]
    # segmentation_maps_batch = [y_chips_padded[_] for _ in range(BATCH_SIZE)]

    # del x_chips_padded, y_chips_padded

    del x_chips_total, y_chips_total

    print('images_batch & segmentation_maps_batch loaded')

    batches = [UnnormalizedBatch(images=images_batch, segmentation_maps=segmentation_maps_batch) for _ in
               range(NB_BATCHES)]

    # seq = simple_seq()
    seq = aug_seq()
    print('seq loaded')

    batches_aug = list(seq.augment_batches(batches, background=False))

    print('augmentation finished')
    if test:
        time_end = time.time()
        print("Complete load & augmentation pipeline done in %.2fs" % (time_end - time_start,))
        # print("Resizing & returning augmented x_trn, y_trn datasets")

    print('resizing img & msk')

    img = np.array([batches_aug[a].images_aug for a in range(NB_BATCHES)]).reshape(amt, ISZ, ISZ, 3)
    msk = (np.array([batches_aug[a].segmentation_maps_aug for a in range(NB_BATCHES)]).reshape(amt, ISZ, ISZ, 1))

    print(img.shape, msk.shape)

    img = np.array(img).reshape(amt, ISZ, ISZ, 3)

    # img = [stretch_n(i) for i in img]

    # print(img.shape)

    img = np.array(img).reshape(amt, ISZ, ISZ, 3)
    print(img.dtype)
    print(img.shape)
    # img = img / 255.

    # img = img.astype('uint8')
    print(msk.dtype)
    print(msk.shape)
    msk = msk.clip(max=1)
    print(msk.dtype)

    return img, msk


