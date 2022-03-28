import image_manipulation as im
import numpy as np
from tqdm.notebook import tqdm
import torch
import torchvision.transforms as transforms


# data transformation 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

# softmax
softy = torch.nn.Softmax(dim=1)

# test network for probabilities of target class
def calc_shuffle_probabilities(network, device, iterations, patches, batch_size, baseline_target, option=[0]):
    option_dict = {
        0: "Shuffle and Test without perturbing any patch",
        1: "Insert black patch",
        2: "Insert mean color patch",
        3: "Insert random other patch",
        4: "Insert most cosine distant patch"
    }
    """
    removed_patch_option:   0 = shuffle and test without removing any patch
                            1 = black
                            2 = mean color of original image
                            3 = random other patch 
                            4 = most cosine distant patch
    """

    # create original image once for later use (e.g. get mean color)
    image = im.reassemble_image(patches)
    patch_length = len(patches[0])
    
    # final lists for storing probabilites for graph
    mean_list = np.zeros((5,len(patches)))
    std_list = np.zeros((5,len(patches)))

    for option in tqdm(option):
        print("Option: {}".format(option_dict[option]))

        # if option 0 --> only do it once
        if option == 0:
            number_of_patches = 1
        else:
            number_of_patches = len(patches)

        # do for every patch
        for position in tqdm(range(number_of_patches)):

            # create list to save probability of this run
            single_prob_list = []

            # create fresh copy of patches to insert black patch
            patches_copy = np.copy(patches)

            # create empty/black patch
            new_patch = np.zeros((patches[0].shape))

            # insert black patch
            if option == 1:
                # insert black patch to copy
                patches_copy[position] = new_patch

            # insert mean color patch
            elif option == 2:
                # get mean color
                mean_color = np.mean(image, axis=(0,1))
                for x in range(0, patch_length):
                    for y in range(0, patch_length):
                        for z in range(0, 3):
                            new_patch[x][y][z] = mean_color[z]
                patches_copy[position] = new_patch

            # insert most cosine distant patch
            elif option == 4:
                flattened_removed_patch = patches[position].flatten()
                flatten_list = []
                # create flattened patch_list
                for index in patches_copy:
                    flatten_list.append(index.flatten())
                cosine_list = []
                # create list of cosine values between patch to remove and every other patch
                for index in flatten_list:
                    cosine_list.append(im.cos_sim(flattened_removed_patch, index))
                # get index of min cosine value of cosine list
                min_cosine_index = np.argmin(cosine_list)
                # insert most distant patch to patch list
                patches_copy[position] = patches[min_cosine_index]

            # loop to test image with network for every position
            with torch.no_grad():

                for index in range(int(np.ceil(iterations/batch_size))):

                    # update batch size if fewer iterations than batch size are left over
                    updated_batch_size = min(iterations - index * batch_size, batch_size)

                    # batch handling for slightly faster computation time
                    batch_torch = torch.zeros(updated_batch_size, image.shape[2], image.shape[0], image.shape[1])

                    for batch in range(updated_batch_size):

                        # extra for random insert, because we need new insertion for every iteration
                        if option == 3:
                            # create copy of copy, otherwise it would be overwritten every iteration
                            patches_copy_copy = np.copy(patches)
                            # random integer from 0 to number of patches
                            random_index = np.random.randint(0, number_of_patches)
                            # copy patch from random position
                            random_patch = np.copy(patches[random_index])
                            # insert random patch at position
                            patches_copy_copy[position] = random_patch
                            patches_copy = patches_copy_copy

                        # create shuffled patches and the order
                        shuffled_patches, order = im.shuffle_patch_list(patches_copy)

                        # reconstruct shuffled image
                        jigsaw = im.reassemble_image(shuffled_patches)
                        transformed_image = torch.DoubleTensor(data_transform(jigsaw)).float()

                        batch_torch[batch] = transformed_image

                        # free up variables so the cache does not fill up
                        del shuffled_patches, order, transformed_image, jigsaw

                    # classify image
                    network_output = network(batch_torch.to(device))

                    # softmax to get correct probability
                    network_output = softy(network_output)

                    # attach probabilities to list
                    for no in network_output:
                        single_prob_list.append(no[baseline_target].item())
                    
                    # free up variables so the cache does not fill up
                    del  network_output

            
            # mean and standard deviaton of this run
            single_mean = np.mean(single_prob_list)
            single_std = np.std(single_prob_list)

            mean_list[option][position] = single_mean
            std_list[option][position] = single_std
    
        if option == 0:
            print("Baseline value without removing any patch:")
            print("Mean probability: ", mean_list[0][0])
            print("Standard deviation: ", std_list[0][0])

    return(mean_list, std_list)

# take second element for sort
def takeSecond(elem):
    return elem[1]

# create sorted list with mean patch probability
def mean_tuple_list(amount_of_patches, mean_list):
    # create list for mean probability
    mean_patch_prob_list = []
    # patch number list
    number_list = []

    # fill values to the lists
    for y in range(amount_of_patches):
        tmp_list = []
        # 4 because of 4 options
        for x in range(4):
            tmp_list.append(mean_list[x+1][y])
        # create list sorted after patch number
        mean_patch_prob_list.append(np.mean(tmp_list))
        number_list.append(y)

    # create tuple list
    tuple_list = list(zip(number_list, mean_patch_prob_list))

    # sort list descendeing based on probability
    sorted_list = sorted(tuple_list, key=takeSecond)

    return(sorted_list, mean_patch_prob_list)

# create feature map from probability values per patch
def create_feature_map(patches, prob_list):
    
    # create empty feature map
    feature_map = np.zeros((224,224,3))

    # normalize list against the maximum
    normalized_list = []
    maximum = max(prob_list)
    for entry in prob_list:
        normalized_list.append(entry / maximum)

    # reverse list to get high values for low percentages
    reversed_list = [1 - i for i in normalized_list]

    patch_length = len(patches[0])
    patch_amount = len(patches)
    patch_size = np.sqrt(patch_amount)

    xindex = 0
    yindex = 0
    # go through number of patches
    for patch in range(0, patch_amount):
        # go through single patch (every pixel)
        for x in range(0, patch_length):
            for y in range(0, patch_length):
                for z in range(0, 3):
                    # copie value to each pixel in each patch
                    feature_map[xindex][yindex][z] = reversed_list[patch]
                yindex += 1
            xindex += 1
            yindex -= patch_length

        if ((patch + 1) % patch_size == 0):
            xindex = 0
            yindex += patch_length

    return(feature_map)
