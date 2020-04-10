import os
import os.path
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IMAGENET_MEAN_BGR = [103.939, 116.779, 123.68]

def load_images(data_path, image_height, image_width, plot=False):
    """
    Read an image in BGR,
    resize to image_height x image_width,
    subtract mean of ImageNet dataset
    """
    # Get a list of images in the folder
    os.chdir(data_path)
    list = glob.glob('*.jpg')
    N_images = len(list)
    
    # Create arrays to store data
    images = np.zeros((N_images, image_height, image_width, 3), dtype = np.float32)
    
    if plot:
        fig = plt.figure(figsize=(15,6))
    
    for i in range(0, N_images):
        # Load image
        image_name = list[i]
        image = cv2.imread(image_name)
        
        if plot:
            # Plot an image
            fig.add_subplot(1, N_images, i+1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')          
            plt.show()
        
        # Resize to image_height x image_width
        images[i, :, :, :] = cv2.resize(image.astype(np.float32),(image_height, image_width))
        
        # Subtract ImageNet mean
        images[i, :, :, :] -= IMAGENET_MEAN_BGR

    return images

def load_images_with_labels(data_path, labels_path, image_height, image_width):
    """
    Read an image in BGR,
    resize to image_height x image_width,
    subtract mean of ImageNet dataset.
    Assign a label to an image:
    1 if there is a tumour, 0 otherwise
    """
    # Get a list of images in the folder
    os.chdir(data_path)
    list = glob.glob('*.jpeg')
    N_images = len(list)
    return N_images
    
    # Create arrays to store data and labels
    images = np.zeros((N_images, image_height, image_width, 3), dtype = np.float32)
    labels = -1 * np.ones((N_images, 1), dtype = np.float32)
        
    for i in range(0, N_images):
        # Load image in BGR
        image_name = list[i]
        image = cv2.imread(image_name)
        # Load image in RGB
        # image = plt.imread(image_name)
        
        # Convert RGB to BGR 
        #image = image[:, :, [2, 1, 0]]

        # Resize to image_height x image_width
        images[i, :, :, :] = cv2.resize(image.astype(np.float32),(image_height, image_width))
        
        # Subtract ImageNet mean
        images[i, :, :, :] -= IMAGENET_MEAN_BGR
        
        # Assign a label to an image: 
        # 1 if there is a tumour, 0 otherwise
        file_path = labels_path + image_name[:-5] + ".txt"
        
        if os.path.isfile(file_path):
            labels[i] = 1
        else:
            labels[i] = 0
 
    return images, labels

def load_images_with_masks(data_path, mask_path, image_height, image_width, binary=False, plot=False):
    """
    Read an image in BGR,
    resize to image_height x image_width,
    subtract mean of ImageNet dataset.
    Read the corresponding binary mask.
    """
    # Get the list of images
    os.chdir(data_path)
    image_list = glob.glob('*.jpg')
    N_images = len(image_list)
    
    # Get the list of masks
    os.chdir(mask_path)
    mask_list = glob.glob('*.jpg')
    
    # Create arrays to store data
    images = np.zeros((N_images, image_height, image_width, 3), dtype = np.float32)
    masks = np.zeros((N_images, image_height, image_width), dtype = np.float32)
    
    if plot:
        fig = plt.figure(figsize=(15,6))
    
    for i in range(0, N_images):
        # Load image
        image_name = image_list[i]
        os.chdir(data_path)
        image = cv2.imread(image_name)
        
        # Resize to image_height x image_width
        images[i, :, :, :] = cv2.resize(image.astype(np.float32),(image_height, image_width))
        
        # Subtract ImageNet mean
        images[i, :, :, :] -= IMAGENET_MEAN_BGR
        
        # Check if there is a mask
        mask_name = image_name[:-4] + '_mask.jpg'
        if mask_name in mask_list:
            os.chdir(mask_path)
            mask = cv2.resize(plt.imread(mask_name).astype(np.float32), (image_height, image_width))
            if binary:
                mask = 0 * (mask < 128.0) + 1 * (mask >= 128.0)
            masks[i, :, :] = mask
            
        if plot:
            # Plot image
            fig.add_subplot(N_images, 2, 2*i+1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')   
            
            # Plot mask
            fig.add_subplot(N_images, 2, 2*i+2)
            plt.imshow(mask)
            plt.axis('off')  
    
    plt.show()
    return images, masks

def split_train_val(data, labels, train_ratio=0.8):
    """
    Split data on training and validation sets
    """
    # Shuffle indeces
    n = len(data)
    indeces = list(range(0, n))
    np.random.shuffle(indeces)
    
    # Create training set
    train_indeces = indeces[:round(train_ratio * n)]
    X_train = data[train_indeces, :, :, :]
    y_train = labels[train_indeces]
    
    # Create validation set
    val_indeces = indeces[round(train_ratio * n):]
    X_val = data[val_indeces, :, :, :]
    y_val = labels[val_indeces]

    print("Training set:", X_train.shape, y_train.shape)
    print("Validation set:", X_val.shape, y_val.shape)
    
    return X_train, y_train, X_val, y_val

def stratified_train_val(data, labels, train_ratio=0.8, balance_classes=False):
    """
    Create stratified training and validation sets for binary data
    """
    # numbers of positive and negative samples in the dataset
    n_pos = int(sum(labels))
    n_neg = data.shape[0] - n_pos
    print('Number of negative samples: ', n_neg)
    print('Number of positive samples: ', n_pos)
    print('Fraction of positive samples: ', n_pos / data.shape[0] * 100, '%')
    
    # to fix class imbalance equalize
    # the numbers of negative and positive samples
    if balance_classes:
        if n_neg > n_pos:
            n_neg = n_pos
        else:
            n_pos = n_neg
    
    # print the numbers of negative/positive samples
    # in training and validation sets
    print('Positive samples:',
          round(train_ratio * n_pos), "in y_train,",
          round((1 - train_ratio) * n_pos), "in y_val")
    print('Negative samples:',
          round(train_ratio * n_neg), "in y_train,",
          round((1 - train_ratio) * n_neg), "in y_val")
    
    # extract, shuffle and split indeces of positive samples
    pos_indeces = (np.where(labels == 1))[0]
    np.random.shuffle(pos_indeces)
    pos_indeces_train = pos_indeces[:round(train_ratio * n_pos)]
    pos_indeces_val = pos_indeces[round(train_ratio * n_pos):]
    
    # extract, shuffle and split indeces of negative samples
    neg_indeces = (np.where(labels == 0))[0]
    np.random.shuffle(neg_indeces)
    neg_indeces_train = neg_indeces[:round(train_ratio * n_neg)]
    neg_indeces_val = neg_indeces[round(train_ratio * n_neg):]
    
    # create a training set
    train_indeces = np.append(pos_indeces_train, neg_indeces_train, axis=0)
    np.random.shuffle(train_indeces)
    X_train = data[train_indeces, :, :, :]
    y_train = labels[train_indeces]

    # create a validation set
    val_indeces = np.append(pos_indeces_val, neg_indeces_val, axis = 0)
    np.random.shuffle(val_indeces)
    X_val = data[val_indeces, :, :, :]
    y_val = labels[val_indeces]

    print("Training set:", X_train.shape, y_train.shape)
    print("Validation set:", X_val.shape, y_val.shape)
    
    return X_train, y_train, X_val, y_val

def standardize(X, train_mean=None, train_sd=None, training_set=False):
    # Standartise data using mean and sd of the training set
    if train_mean == None:
        train_mean = np.mean(X)
        train_sd = np.std(X, ddof = 1)
    X_std = (X - train_mean) / train_sd
    print('This set had mean', np.mean(X), 'and s.d.', np.std(X, ddof = 1))
    print('Standardized set has mean', np.mean(X_std), 'and s.d.', np.std(X_std, ddof = 1))
    if training_set:
        return X_std, train_mean, train_sd
    else:
        return X_std
