from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint

import warnings

import rasterio
from rasterio.windows import Window
import geopandas as gpd
from rasterio.features import rasterize

from collections import defaultdict
import random

from PIL import Image

from pystac import (Catalog, CatalogType, Item, Asset, LabelItem, Collection)
from shapely.geometry import box, Point, Polygon, MultiPolygon

from skimage.util.shape import view_as_windows
from skimage import measure

import cv2

from pycocotools import mask as pycoco_Mask

import secrets
import os

import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.batches import UnnormalizedBatch

import json

!pip install cython
!pip install pycocotools

train1_cat = Catalog.from_file('/storage/Open_Cities_AI_Challenge/Data/train_tier_1/catalog.json')

collections = {cols.id:cols for cols in train1_cat.get_children()}

def get_dict_of_collections():
    ids = defaultdict(list)
    ids_labels = defaultdict(list)
    for a in collections:
        for i in collections[a].get_all_items():
            if 'label' not in i.id:
                ids[a].append(i.id)
            else:
                ids_labels[a].append(i.id)
    return ids, ids_labels

ids, ids_labels = get_dict_of_collections()

def get_list_of_collections_no_labels():
    ids_list = []
    for a in collections:
        for i in collections[a].get_all_items():
            if 'label' not in i.id:
                ids_list.append((a, i.id))
    return ids_list

def get_list_of_collections_full():
    ids_list = []
    for a in collections:
        for i in collections[a].get_all_items():
            ids_list.append((a, i.id))
    return ids_list

ids_list = get_list_of_collections_no_labels()

def build_out_href_ids():
    for a in ids:
        for i in collections[str(a)].get_all_items():
            #print(i.id)
            #pprint(i.properties)
            if 'label' in i.id:
                gpd.read_file(i.make_asset_hrefs_absolute().assets['labels'].href)
                pass
            else:
                #print('raster metadata:')
                rasterio.open(i.make_asset_hrefs_absolute().assets['image'].href).meta

build_out_href_ids()


def get_rasters_labels_and_image():
    raster_and_label_and_images_list = defaultdict(list)
    for a in ids:
        for b in ids[a]:
            image = collections[str(a)].get_item(id=str(b))
            labels = collections[str(a)].get_item(id=str(b) + '-labels')
            # labels_gdf = gpd.read_file(labels.assets['labels'].href)
            raster = rasterio.open(image.assets['image'].href)

            # print(raster.res)
            raster_and_label_and_images_list[a].append([b, raster, labels, image])

    return raster_and_label_and_images_list

def center(minx, miny, maxx, maxy):
    center_x = (maxx+minx)/2
    center_y = (maxy+miny)/2
    return(Point(center_x, center_y))

def set_random_center_within_poly(x_max, y_max, s, image_poly, raster, y_labels_gdf):
    is_center = False
    mask_threshold_met = False
    mask_ratio_met = False
    while not is_center:
        while not mask_threshold_met:
            x_sample, y_sample = random.randrange(0, x_max), random.randrange(0, y_max)
            test_window = Window(x_sample, y_sample, s, s)
            test_box = box(*rasterio.windows.bounds(test_window, raster.meta['transform']))
            test_box_gdf = gpd.GeoDataFrame(geometry=[test_box], crs=raster.meta['crs'])
            with warnings.catch_warnings():
            # ignore all caught warnings
                warnings.filterwarnings("ignore")
            # execute code that will generate warnings
                test_box_gdf = test_box_gdf.to_crs({'init':'epsg:4326'})
            test_chip = gpd.sjoin(y_labels_gdf, test_box_gdf, how='inner', op='intersects')
            test_chip_shapes = [(geom, 255) for geom in test_chip.geometry]
            if len(test_chip_shapes) > 3:
                mask_threshold_met = True
            else:
                break
            minx, miny, maxx, maxy = [test_box_gdf.bounds[a][0] for a in test_box_gdf.bounds]
            center_point = center(minx, miny, maxx, maxy)
            is_center = center_point.within(image_poly)
            if mask_threshold_met and not is_center:
                mask_threshold_met = False
    return(x_sample, y_sample)


def aug_seq():
    seq = iaa.Sequential([
        iaa.Fliplr(.5),  # horizontal flips
        iaa.Flipud(.5),  # vertical flips
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-35, 35),
            shear=(-10, 10)),

    ], random_order=True)  # apply augmenters in random order
    return seq


def window_creation(chips_x, chips_y, ISZ):
    dim_Max = chips_x.shape[0]
    windows = dim_Max // ISZ
    x_windows = view_as_windows(chips_x, (ISZ, ISZ, 3), ISZ)[:windows, :windows, 0, :, :]
    y_windows = view_as_windows(chips_y, (ISZ, ISZ), ISZ)[:windows, :windows, :, :]
    return x_windows, y_windows, windows


def chip_verification(x_windows, y_windows, dim_Max, ISZ, min_label_coverage, max_label_coverage):
    images_aug = aug_seq()

    x_chips_verified = []
    y_chips_verified = []
    for x in range(dim_Max):
        for y in range(dim_Max):
            label_coverage = np.count_nonzero(y_windows[x][y]) / (ISZ * ISZ)
            if label_coverage >= min_label_coverage and label_coverage <= max_label_coverage:
                x_aug, y_aug = images_aug(image=x_windows[x][y], segmentation_maps=
                np.expand_dims(np.expand_dims(
                    np.array(y_windows[x][y]), axis=3), axis=0))

                # print(x_aug.shape, y_aug.shape)

                x_chips_verified.append(x_aug)
                x_chips_verified.append(x_windows[x][y])
                y_chips_verified.append(np.squeeze(y_aug))
                y_chips_verified.append(y_windows[x][y])

                # x_chips_verified.append(x_windows[x][y])
                # y_chips_verified.append(y_windows[x][y])

    return np.array(x_chips_verified), np.expand_dims(np.array(y_chips_verified), axis=3)

def mask_to_polygons(mask, epsilon=.01, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    # image, (remove from original)
    contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def group_of_polygons(polygon_group, im_size):
    polygon_mask_group = []
    polygons = [polygon_group.geoms[a] for a in range(len(polygon_group.geoms))]
    #polygons = list(polygons)
    for _ in polygons:
        img_mask = np.zeros([im_size, im_size], np.uint8)
        if not polygons:
            return img_mask
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exterior = int_coords(_.exterior.coords)
        interior = [int_coords(pi.coords) for pi in _.interiors]
        if len(interior) == 0:
            cv2.fillConvexPoly(img_mask, exterior, 1)
            cv2.fillPoly(img_mask, interior, 0)
        else:
            cv2.fillConvexPoly(img_mask, exterior, 1)
            cv2.fillPoly(img_mask, interior, 0)
        polygon_mask_group.append(img_mask)
    return polygon_mask_group

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0.5 if i <= 0 else i for i in segmentation]
        #print('segmentation:', segmentation)
        polygons.append([segmentation]) # x,y pairs in sequence, no sub-grouping, [] for each polygon mask

    return polygons

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

folder_dir = '/storage/Open_Cities_AI_Challenge/Data/train_tier_1/Mask_RCNN Dataset/'


def create_meta_image(S=4096, test=True, ISZ=512):  # , coco_output_train = {}, coco_output_val = {}):

    s = S

    raster_labels_and_images_list = get_rasters_labels_and_image()

    added_round_2 = [('znz', 'aee7fd')]

    dataset_labels = [('znz', 'bd5c14'), ('nia', '825a50'), ('znz', '06f252'), ('znz', '076995'), ('znz', '33cae6'),
                     ('kam', '4e7c7f'), ('znz', 'bc32f1'), ('znz', 'c7415c'), ('znz', '75cdfa'),
                     ('mon', 'f15272'), ('acc', 'd41d81'), ('acc', 'ca041a'), ('znz', '3f8360'), ('mon', '493701'),
                     ('dar', 'b15fce'), ('acc', 'a42435'), ('dar', '42f235')]

    no_initial_results = [('znz', 'aee7fd')]

    broken_window_size = [('znz', '9b8638')]  # This is the broken window size dataset

    # This is a collection of very few buildings, or has sub-sets which are very poorly labeled
    ids_maybe_round_two = [('znz', 'e52478'), ('dar', '353093'), ('znz', '425403'), ('znz', '3b20d4')]
    ids_need_manual_selection = [('dar', '0a4c40')]
    ids_high_zoom = [('acc', '665946')]  # Not included
    ids_low_zoom = [('ptn', 'abe1a3'), ('ptn', 'f49f31')]  # Also not included

    image_id = 0
    segmentation_id = 0

    INFO = {
        "description": "Building Dataset - DrivenData Open Cities AI Challenge:\
    Segmenting Buildings for Disaster Resilience",
        "url": "",
        "version": "0.2.0",
        "year": 2020,
        "contributor": "risuo",
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'buiding',
            'supercategory': 'buildings',
        },
    ]

    coco_output_val = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    coco_output_trn = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    for a in dataset_labels:
        print(a)
        for img_indx, item in enumerate(raster_labels_and_images_list[str(a[0])]):
            if item[0] == a[1]:

                raster = raster_labels_and_images_list[str(a[0])][img_indx][1]
                y_labels = raster_labels_and_images_list[str(a[0])][img_indx][2]
                image = raster_labels_and_images_list[str(a[0])][img_indx][3]
                y_labels_gdf = gpd.read_file(y_labels.assets['labels'].href)
                x_max = raster.meta['width']
                y_max = raster.meta['height']
                print('max dimensions:', x_max, y_max)
                x_min = 0
                y_min = 0
                strides_x = (x_max // s) + 1
                strides_y = (y_max // s) + 1
                print('strides:', strides_x, strides_y)

                for i in range(strides_x):
                    for j in range(strides_y):
                        # print('stride:', i, j)
                        x_sample, y_sample = (x_min + (s * i)), (y_min + (s * j))
                        test_window = Window(x_sample, y_sample, s, s)
                        test_box = box(*rasterio.windows.bounds(test_window, raster.meta['transform']))
                        test_box_gdf = gpd.GeoDataFrame(geometry=[test_box], crs=raster.meta['crs'])
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            test_box_gdf = test_box_gdf.to_crs({'init': 'epsg:4326'})
                        test_chip = gpd.sjoin(y_labels_gdf, test_box_gdf, how='inner', op='intersects')
                        test_chip_shapes = [(geom, 255) for geom in test_chip.geometry]

                        if len(test_chip_shapes) == 0:
                            break
                        else:
                            x_offset, y_offset = x_sample, y_sample
                            window = Window(x_offset, y_offset, s, s)
                            win_box = box(*rasterio.windows.bounds(window, raster.meta['transform']))
                            win_box_gdf = gpd.GeoDataFrame(geometry=[win_box], crs=raster.meta['crs'])

                            with warnings.catch_warnings():
                                # ignore all caught warnings
                                warnings.filterwarnings("ignore")
                                # execute code that will generate warnings
                                win_box_gdf = win_box_gdf.to_crs({'init': 'epsg:4326'})

                            win_arr = raster.read(window=window)
                            win_arr = np.moveaxis(win_arr, 0, 2)

                            x_out = win_arr[:s, :s, 0:3]

                            gdf_chip = gpd.sjoin(y_labels_gdf, win_box_gdf, how='inner', op='intersects')
                            burn_val = 255
                            shapes = [(geom, burn_val) for geom in gdf_chip.geometry]
                            chip_tfm = rasterio.transform.from_bounds(*win_box_gdf.bounds.values[0], s, s)

                            labels_array_stacked = rasterize(shapes, (s, s), transform=chip_tfm, dtype='uint8')

                            y_out = labels_array_stacked[:s, :s]

                            x_windows, y_windows, dim_Max = window_creation(x_out, y_out, ISZ)
                            del x_out, y_out

                            x_chips_verified, y_chips_verified = chip_verification(x_windows, y_windows, dim_Max, ISZ,
                                                                                   min_label_coverage=.10,
                                                                                   max_label_coverage=.90)

                            print(x_chips_verified.shape, y_chips_verified.shape)

                            # for _ in range(5):
                            #    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
                            #    ax1.imshow(x_chips_verified[_])
                            #    ax2.imshow(x_chips_verified[_])
                            #    ax2.imshow(np.squeeze(y_chips_verified[_]), alpha=0.5)
                            #    plt.show()
                            #    plt.close()
                            # return

                            del x_windows, y_windows

                            if len(x_chips_verified) < 1:
                                break

                            print('final shape x_chips_verified, y_chips_verified:', x_chips_verified.shape,
                                  y_chips_verified.shape)
                            print(len(x_chips_verified))

                            coco_output_trn, coco_output_val, image_id, segmentation_id = convert_dataset_to_files(
                                x_chips_verified, y_chips_verified,
                                coco_output_trn, coco_output_val,
                                image_id, segmentation_id, test, ISZ)
                            # print(image_id)
                            # print(coco_output["annotations"])
                            with open('/storage/coco json files/instances_shape_train_' + str(a[0]) + '_' + str(
                                    a[1]) + '.json', 'w') as output_json_file_1:
                                json.dump(coco_output_trn, output_json_file_1)
                            with open('/storage/coco json files/instances_shape_val_' + str(a[0]) + '_' + str(
                                    a[1]) + '.json', 'w') as output_json_file_2:
                                json.dump(coco_output_val, output_json_file_2)


def create_image_info(image_id, file_name, image_size):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1]
    }

    return image_info

def train_val_split(x_chips_verified, y_chips_verified):
    threshold = int((len(x_chips_verified) * .85) // 1)
    x_trn, y_trn = x_chips_verified[:threshold], y_chips_verified[:threshold]
    x_val, y_val = x_chips_verified[threshold:], y_chips_verified[threshold:]
    return x_trn, y_trn, x_val, y_val


def convert_dataset_to_files(x_chips_verified, y_chips_verified, coco_output_trn, coco_output_val, image_id,
                             segmentation_id, test, ISZ):
    image_id = image_id
    segmentation_id = segmentation_id
    coco_output_trn = coco_output_trn
    coco_output_val = coco_output_val
    ISZ = ISZ
    print('Converting:', len(x_chips_verified), ':files.')

    x_trn, y_trn, x_val, y_val = train_val_split(x_chips_verified, y_chips_verified)

    for _ in range(len(x_trn)):
        x_file = x_trn[_]
        y_file = y_trn[_]
        name = secrets.token_urlsafe(32)

        x_file = Image.fromarray(x_file)
        # Formatted for pre-set Google Drive Folder Structure
        image_loc_id = '/content/drive/My Drive/Detectron2 Datasets/OCAI Building Segmentation/' + 'train' + '/' + name + '.png'
        x_file.save('/storage/output_train_files/' + name + '.png')

        image_info = create_image_info(image_id, image_loc_id, (ISZ, ISZ))
        coco_output_trn["images"].append(image_info)


        msk = y_file.clip(max=1)
        msk = mask_to_polygons(msk)
        msk = group_of_polygons(msk, 512)

        for mask in msk:
            mask_coordinates = binary_mask_to_polygon(np.squeeze(mask), tolerance=0)
            segmentation_id = segmentation_id
            class_id = 1
            is_crowd = 0
            binary_mask = resize_binary_mask(mask, (ISZ, ISZ))
            segmentation = mask_coordinates[0]
            binary_mask_encoded = pycoco_Mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
            area = pycoco_Mask.area(binary_mask_encoded)
            bounding_box = pycoco_Mask.toBbox(binary_mask_encoded)

            annotation_info = {
                "id": segmentation_id,
                "image_id": image_id,
                "category_id": class_id,
                "iscrowd": is_crowd,
                "area": area.tolist(),
                "bbox": bounding_box.tolist(),
                "segmentation": segmentation,
                "width": binary_mask.shape[1],
                "height": binary_mask.shape[0],
            }

            coco_output_trn["annotations"].append(annotation_info)

            segmentation_id += 1
        image_id += 1

    for _ in range(len(x_val)):
        x_file = x_val[_]
        y_file = y_val[_]
        name = secrets.token_urlsafe(32)

        x_file = Image.fromarray(x_file)
        # Formatted for pre-set Google Drive Folder Structure
        image_loc_id = '/content/drive/My Drive/Detectron2 Datasets/OCAI Building Segmentation/' + 'val' + '/' + name + '.png'
        x_file.save('/storage/output_val_files/' + name + '.png')

        image_info = create_image_info(image_id, image_loc_id, (ISZ, ISZ))
        coco_output_val["images"].append(image_info)

        msk = y_file.clip(max=1)
        msk = mask_to_polygons(msk)
        msk = group_of_polygons(msk, 512)

        for mask in msk:
            mask_coordinates = binary_mask_to_polygon(np.squeeze(mask), tolerance=0)
            segmentation_id = segmentation_id
            class_id = 1
            is_crowd = 0
            binary_mask = resize_binary_mask(mask, (ISZ, ISZ))
            segmentation = mask_coordinates[0]
            binary_mask_encoded = pycoco_Mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
            area = pycoco_Mask.area(binary_mask_encoded)
            bounding_box = pycoco_Mask.toBbox(binary_mask_encoded)

            annotation_info = {
                "id": segmentation_id,
                "image_id": image_id,
                "category_id": class_id,
                "iscrowd": is_crowd,
                "area": area.tolist(),
                "bbox": bounding_box.tolist(),
                "segmentation": segmentation,
                "width": binary_mask.shape[1],
                "height": binary_mask.shape[0],
            }

            coco_output_val["annotations"].append(annotation_info)

            segmentation_id += 1
        image_id += 1

    return coco_output_trn, coco_output_val, image_id, segmentation_id

create_meta_image(S = 4096, test = True, ISZ = 512)


