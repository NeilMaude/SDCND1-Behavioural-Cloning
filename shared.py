from sklearn.utils import shuffle

import numpy as np
from scipy.misc import imread, imresize

# Helper functions for pre-processing
# Resize the images
def resize(imgs, shape=(32, 16, 3)):
    # Using 32*16 - downsampling by 10x each dimension
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = imresize(img, shape)
    return imgs_resized

# Grayscale the images
def grayscale(imgs):
    # Simple averaging process
    return np.mean(imgs, axis=3, keepdims=True)

# Normalise the image pixel values
def normalise(imgs):
    return imgs / (255.0 / 2) - 1

# Pre-processing function - takes a set of images, sizes, greyscales and normalises...
def preprocess(imgs):
    imgs_processed = resize(imgs)
    imgs_processed = grayscale(imgs_processed)
    imgs_processed = normalise(imgs_processed)
    return imgs_processed

# Read images from files
def read_imgs(img_paths):
    imgs = np.empty([len(img_paths), 160, 320, 3])      # assumes 320*160 image size, RGB
    for i, path in enumerate(img_paths):
        imgs[i] = imread(path)
    return imgs

# Randomly flip some images on the x-axis
def random_flip(imgs, labels):
    new_imgs = np.empty_like(imgs)
    new_labels = np.empty_like(labels)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        if np.random.choice(2):
            # chose to flip this image
            new_imgs[i] = np.fliplr(img)
            new_labels[i] = label * -1
        else:
            # did not choose to flip this image
            new_imgs[i] = img
            new_labels[i] = label

    return new_imgs, new_labels

# Augment the data set by calling the random_flip() function
def augment(imgs, labels):
    imgs_augmented, labels_augmented = random_flip(imgs, labels)
    return imgs_augmented, labels_augmented

# Generator for random batches
def generate_batches(imgs, labels, batch_size):
    #Generates random batches of the input data.
    # imgs: The input images. X_train
    # labels: The ground-truth steering angles associated with each image.  y_train
    # batch_size: The size of each minibatch.
    # yield: A tuple (images, angles), where both images and angles have batch_size elements.
    num_imgs = len(imgs)
    while True:
        indices = np.random.choice(num_imgs, batch_size)
        batch_imgs_raw, labels_raw = read_imgs(imgs[indices]), labels[indices].astype(float)
        batch_imgs, batch_labels = augment(preprocess(batch_imgs_raw), labels_raw)
        yield batch_imgs, batch_labels

# Split up an array of images and labels into a training and validation set
def train_validation_split(image_array, label_array, train_percent=0.8):
    n = int(len(image_array) * train_percent)
    i_array, l_array = shuffle(image_array, label_array)
    train_i = i_array[:n]
    valid_i = i_array[n:]
    train_l = l_array[:n]
    valid_l = l_array[n:]
    return train_i, train_l, valid_i, valid_l