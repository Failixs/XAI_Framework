import image_manipulation as im
import numpy as np
import torch
from tqdm.notebook import tqdm
from captum.attr import visualization as viz
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

# DAUC with different replacement strategies
def patch_dauc(patches_attribution, patches_original, network, device, option=[0], visualization=False, show_final_image=False):

    image = im.reassemble_image(patches_original)
    patches_attribution = np.abs(patches_attribution)
    # find target class
    network_output = get_network_output(image, network, device)
    # get top prediction probability and class
    topk_prob, topk_label = torch.topk(network_output, 1000)
    target = topk_label[0][0]
    class_prob = topk_prob[0][0]
    print("Calculate baseline:")
    print('The selected image is from class {} with {}%'.format(target.item(), class_prob * 100))

    # create flattened patch_list
    flatten_list = []
    for k in patches_original:
        flatten_list.append(k.flatten())
   
    # make black patch
    patch_shape = patches_original[0].shape
    black_patch = np.zeros(patch_shape)

    # make mean colored patch
    meancolor_patch = np.full(patch_shape, np.mean(image, axis=(0,1)))

    # length of one patch
    patch_length = len(patches_original[0])
    number_of_patches = len(patches_original)

    # probability list; start the list with original probability, 5 options
    prob_list = np.zeros((5, number_of_patches + 1))
    for index in range (5):
        prob_list[index][0] = class_prob

    with torch.no_grad():
        # empty cache
        torch.cuda.empty_cache()

        # do for every option
        for option in tqdm(option, desc="{} Options".format(len(option))):
            
            # do this part only if option is not 0 
            if option != 0:

                # create copy of original attribution patches
                patches_attribution_copy = np.copy(patches_attribution)

                # create copy of patches from original image
                copy_of_patches_original = np.copy(patches_original)

                # do for every patch insertion and network probability
                for index in tqdm(range(number_of_patches), desc="Option {}".format(option)):
                    # get index of highest patch sum
                    index_of_max_sum = get_index_of_highest_patch_sum(patch_length, number_of_patches, patches_attribution_copy)

                    # insert -2 in max sum patch to get it out of the equation next round
                    # (nothing will be smaller if smallest values are -1)
                    patches_attribution_copy[index_of_max_sum] = np.full(patches_attribution[0].shape, -2)

                    # insert black patch
                    if option == 1:
                        copy_of_patches_original[index_of_max_sum] = black_patch

                    # insert meancolored patch
                    elif option == 2:
                        copy_of_patches_original[index_of_max_sum] = meancolor_patch

                    # insert random other patch
                    elif option == 3:
                        # random integer from 0 to number of patches
                        random_index = np.random.randint(0, number_of_patches)
                        copy_of_patches_original[index_of_max_sum] = patches_original[random_index]

                    elif option == 4:
                        # flatten the patch to remove (to compare it, it needs to be flattened)
                        flattened_removed_patch = patches_original[index_of_max_sum].flatten()
                        
                        # create list of cosine values between patch to remove and every other patch
                        cosine_list = []
                        for k in flatten_list:
                            cosine_list.append(im.cos_sim(flattened_removed_patch, k))

                        # get index of min cosine value of cosine list
                        min_cosine_index = np.argmin(cosine_list)

                        # exchange patch with most distance patch according to cosine similarity
                        copy_of_patches_original[index_of_max_sum] = patches_original[min_cosine_index]

                    # reassembling for output visualization and network test
                    reassembled_image = im.reassemble_image(copy_of_patches_original)

                    # test network for probability
                    network_output = get_network_output(reassembled_image, network, device)

                    # get target class probability
                    prob = network_output[0][target].item()
                    
                    # insert probability to list
                    prob_list[option][index+1] = prob

                    if visualization == True:
                        # visualization
                        print("Patch {}:".format(index+1))
                        _ = viz.visualize_image_attr(reassembled_image, reassembled_image, 
                                                    method="original_image",
                                                    sign="all",show_colorbar=False, title="")
            
            # just for visualization of option 0
            else:
                reassembled_image = np.copy(image)

            # show image end result
            if show_final_image == True and visualization == False:
                _ = viz.visualize_image_attr(reassembled_image, reassembled_image, 
                                                        method="original_image",
                                                        sign="all",show_colorbar=False, title="") 

    return(prob_list)

# IAUC with different starting images
def patch_iauc(patches_attribution, patches_original, network, target, device, option=[1], visualization=False):

    image = im.reassemble_image(patches_original)
    image_shape = image.shape
    patches_attribution = np.abs(patches_attribution)

    # create flattened patch_list
    flatten_list = []
    for k in patches_original:
        flatten_list.append(k.flatten())

    # make black patch
    patches_shape = patches_original.shape

    # length of one patch
    patch_length = len(patches_original[0])
    number_of_patches = len(patches_original)

    # probability list
    prob_list = np.zeros((5, number_of_patches + 1))

    with torch.no_grad():
        # empty cache
        torch.cuda.empty_cache()

        # do for every option
        for option in tqdm(option, desc="{} Options".format(len(option)-1)):
                
            # create baseline image
            baseline_image = np.zeros(image_shape)
            baseline_patches = np.zeros(patches_shape)

            # option 0 makes no sense here, but for consistency with other options it stays
            if option != 0:

                # black image
                if option == 1:
                    baseline_image = baseline_image

                # mean colored image
                elif option == 2:
                    baseline_patches = np.full(patches_shape, np.mean(image, axis=(0,1)))
                    baseline_image = np.full(image_shape, np.mean(image, axis=(0,1)))
                
                # random patched image
                elif option == 3:
                    for index in range(number_of_patches):
                        # random integer from 0 to number of patches
                        random_index = np.random.randint(0, number_of_patches)
                        baseline_patches[index] = patches_original[random_index]
                    baseline_image = im.reassemble_image(baseline_patches)

                # most cosine distant image
                elif option == 4:
                    for index in range(number_of_patches):

                        # flatten patch
                        flattened_patch = patches_original[index].flatten()

                        # create list of cosine values between patch to remove and every other patch
                        cosine_list = []
                        for k in flatten_list:
                            cosine_list.append(im.cos_sim(flattened_patch, k))

                        # get index of min cosine value of cosine list
                        min_cosine_index = np.argmin(cosine_list)

                        # exchange patch with most distance patch according to cosine similarity
                        baseline_patches[index] = np.copy(patches_original[min_cosine_index])
                    baseline_image = im.reassemble_image(baseline_patches)

                if visualization == True:
                        # visualization
                        print("Patch 0:")
                        _ = viz.visualize_image_attr(baseline_image, baseline_image, 
                                                    method="original_image",
                                                    sign="all",show_colorbar=False, title="")

                # get baseline probability
                # test network for probability
                network_output = get_network_output(baseline_image, network, device)

                # get target class probability
                prob = network_output[0][target].item()
                
                # insert baseline probability to list
                prob_list[option][0] = prob

                # create copy of original attribution patches
                patches_attribution_copy = np.copy(patches_attribution)

                # do for every patch insertion and network probability
                for index in tqdm(range(number_of_patches), desc="Option {}".format(option)):
                    # get index of highest patch sum
                    index_of_max_sum = get_index_of_highest_patch_sum(patch_length, number_of_patches, patches_attribution_copy)

                    # insert -2 in max sum patch to get it out of the equation next round
                    # (nothing will be smaller if smallest values are -1)
                    patches_attribution_copy[index_of_max_sum] = np.full(patches_attribution[0].shape, -2)

                    # insert real patch
                    baseline_patches[index_of_max_sum] = np.copy(patches_original[index_of_max_sum])

                    # reassembling for output visualization and network test
                    reassembled_image = im.reassemble_image(baseline_patches)

                    # test network for probability
                    network_output = get_network_output(reassembled_image, network, device)

                    # get target class probability
                    prob = network_output[0][target].item()
                    
                    # insert probability to list
                    prob_list[option][index+1] = prob

                    if visualization == True:
                        # visualization
                        print("Patch {}:".format(index+1))
                        _ = viz.visualize_image_attr(reassembled_image, reassembled_image, 
                                                    method="original_image",
                                                    sign="all",show_colorbar=False, title="")
            else:
                print("Option 0 not available.")

    return(prob_list)
        
# get index of highest patch sum
def get_index_of_highest_patch_sum(patch_length, number_of_patches, patches_attribution):
    stored_sum = -(patch_length**2)
    index_of_max_sum = 0
    for counter in range(number_of_patches):
        tmp_array = patches_attribution[counter].flatten()
        tmp_sum = np.sum(tmp_array)
        if (tmp_sum > stored_sum):
            stored_sum = tmp_sum
            index_of_max_sum = counter
    return(index_of_max_sum)
        
# get network output
def get_network_output(image, network, device):
    image_to_explain = data_transform(image)
    # get network output and softmax it
    network_output = network(image_to_explain.to(device).reshape(-1,3,224,224).float())
    network_output = softy(network_output)
    return(network_output)
