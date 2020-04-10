import numpy as np
import cv2
import glob
import itertools
from keras.preprocessing.image import ImageDataGenerator


def getImageArr(path, width, height, imgNorm="sub_mean", odering='channels_first'):
    try:
        # img = cv2.imread(path, 1)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img / 255.0
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
            # print('shape 0: ', np.shape(img))
        # print('shape 1: ', np.shape(img))
        return img
    except (Exception):
        print(path)
        img = np.zeros((height, width, 3))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
            # print('shape 2: ', np.shape(img))
        # print('shape 3: ', np.shape(img))
        return img
    

def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        #Sidong's Dataset, do not need resize
        img = cv2.resize(img, (width, height))
        #commented in GrayScale masks
        imgB = img[:, :, 0]
        imgG = img[:, :, 1]
        imgR = img[:, :, 2]

        #Sidong's Dataset
        #threshold=100
        #Bladder Dataset
        threshold=0
        seg_labels[:, :, 0] = ((imgR <= threshold) & (imgG <= threshold) & (imgB <= threshold)).astype(int)
        if (nClasses==2):
            seg_labels[:, :, 1] = ((imgR > threshold) | (imgG > threshold) | (imgB > threshold)).astype(int)
        elif (nClasses==4):
            seg_labels[:, :, 1] = ((imgR > threshold) | (imgG < threshold) | (imgB < threshold)).astype(int)
            seg_labels[:, :, 2] = ((imgR < threshold) | (imgG > threshold) | (imgB < threshold)).astype(int)
            seg_labels[:, :, 3] = ((imgR < threshold) | (imgG < threshold) | (imgB > threshold)).astype(int)
        else:
            #other datasets
            for c in range(nClasses):
                seg_labels[:, :, c] = (img == c).astype(int)

    except Exception:
        print (Exception)

    #seg_labels = np.reshape(seg_labels, (width * height, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, 
                               input_height, input_width, 
                               output_height, output_width):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        assert (im.split('\\')[-1].split(".")[0] == seg.split('\\')[-1].split(".")[0])

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)
        

def imageSegmentationArray(images_path, segs_path, batch_size, n_classes, 
                           input_height, input_width, 
                           output_height, output_width):
    
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    X = []
    y = []

    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        assert (im.split('\\')[-1].split(".")[0] == seg.split('\\')[-1].split(".")[0])

        X.append(getImageArr(im, input_width, input_height))
        y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

    X = np.array(X)
    y = np.array(y)

    return X, y


def augmentedDataGenerator(images_path, segs_path, batch_size, n_classes, 
                           input_height, input_width, 
                           output_height, output_width):
    
    X, y = imageSegmentationArray(images_path, segs_path, batch_size, n_classes, 
                                  input_height, input_width, 
                                  output_height, output_width)
    
    seed = 42
    rotation_range = 360

    image_datagen = ImageDataGenerator(rotation_range=rotation_range,
                                       horizontal_flip=True,
                                       vertical_flip=True, 
                                       data_format='channels_first')

    mask_datagen = ImageDataGenerator(rotation_range=rotation_range,
                                      horizontal_flip=True,
                                      vertical_flip=True, 
                                      data_format='channels_last')

    image_generator = image_datagen.flow(X,
                                         batch_size=batch_size, 
                                         seed=seed)

    mask_generator = mask_datagen.flow(y, 
                                       batch_size=batch_size, 
                                       seed=seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    
    return train_generator