import numpy as np
from numpy import dot
from numpy.linalg import norm
import random

# cut image to patches list
def make_patches(image, patch_size):

    # define some variables
    number_of_patches = patch_size**2
    patch_length = int(len(image) / patch_size)
    image_length = len(image)

    # create empty patch list of right size
    patch_list = np.zeros((number_of_patches, patch_length, patch_length, 3))

    # index for patch list insertion
    patch_index = 0

    # starting x coordinate for slicing
    for x in range(0, image_length, patch_length):
        # starting y coordinate for slicing
        for y in range(0, image_length, patch_length):
            # if x and y are in between the borders of the original image
            if (x + patch_length) <= image_length and (y + patch_length) <= image_length:
                
                # take correct slice and copy it to the patch list
                patch_list[patch_index] = image[y:y+patch_length,x:x+patch_length,:]

                patch_index += 1

    return(patch_list)

# reassemble image from patchlist
def reassemble_image(patch_list):

    patch_length = len(patch_list[0])
    assembled_image = np.zeros((224, 224, 3))
    image_length = len(assembled_image)

    # index for patch list insertion
    patch_index = 0
    
    # starting x coordinate for slicing
    for x in range(0, image_length, patch_length):
        # starting y coordinate for slicing
        for y in range(0, image_length, patch_length):
            # if x and y are in between the borders of the original image
            if (x + patch_length) <= image_length and (y + patch_length) <= image_length:
                
                # take correct slice and copy it to new image
                assembled_image[y:y+patch_length,x:x+patch_length,:] = patch_list[patch_index]

                patch_index += 1

    return assembled_image

# shuffle patch list
def shuffle_patch_list(patch_list):
    # create list to shuffle like patches for later reconstruction of image
    order_list = np.arange(len(patch_list))

    # create random seed for shuffling
    SEED = random.randint(1, 999999999)

    # shuffle order list
    np.random.seed(SEED)
    np.random.shuffle(order_list)

    # shuffle patch list
    np.random.seed(SEED)
    np.random.shuffle(patch_list)

    return patch_list, order_list

# backshuffle image from shuffled patch_list and order_list
def sort_patch_list(patch_list_shuffled, order_list):

    # create list with same size as shuffled list
    patch_list_ordered = np.zeros((patch_list_shuffled.shape))

    # fill empty list with correct order saved in order_list
    for index, originalIndex in enumerate(order_list):
        patch_list_ordered[originalIndex] = patch_list_shuffled[index]
    
    return patch_list_ordered

# cosine similarity
def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))