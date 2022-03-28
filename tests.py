import image_manipulation as im
import feature_map as fm
import evaluation as ev
import numpy as np
import random


def test_make_patches1():

    image = np.zeros((224,224,3))
    patches = np.zeros((4,112,112,3))
    patches_from_function = im.make_patches(image, 2)

    assert patches_from_function.shape == patches.shape, "Should be same dimension (4,112,112,3)"

def test_make_patches2():

    image = np.zeros((224,224,3))
    patches = np.zeros((16,56,56,3))
    patches_from_function = im.make_patches(image, 4)

    assert patches_from_function.shape == patches.shape, "Should be same dimension (16,56,56,3)"

def test_make_patches3():

    image = np.zeros((224,224,3))
    patches = np.zeros((64,28,28,3))
    patches_from_function = im.make_patches(image, 8)

    assert patches_from_function.shape == patches.shape, "Should be same dimension (64,28,28,3)"

def test_reassemble_image1():

    image = np.zeros((224,224,3))
    patches = np.zeros((4,112,112,3))
    image_from_function = im.reassemble_image(patches)

    assert image_from_function.shape == image.shape, "Should be same dimension (224,224,3)"

def test_reassemble_image2():

    image = np.zeros((224,224,3))
    patches = np.zeros((16,56,56,3))
    image_from_function = im.reassemble_image(patches)

    assert image_from_function.shape == image.shape, "Should be same dimension (224,224,3)"

def test_reassemble_image3():

    image = np.zeros((224,224,3))
    patches = np.zeros((64,28,28,3))
    image_from_function = im.reassemble_image(patches)

    assert image_from_function.shape == image.shape, "Should be same dimension (224,224,3)"

def test_cosine_similarity():

    a = [5,0,3,0,2,0,0,2,0,0]
    b = [5,0,3,0,2,0,0,2,0,0]
    cs = im.cos_sim(a,b)

    assert cs == 1.0, "Should be 1.0"

if __name__ == "__main__":
    test_make_patches1()
    print("Passed test: make_patches1()")

    test_make_patches2()
    print("Passed test: make_patches2()")

    test_make_patches3()
    print("Passed test: make_patches3()")

    test_reassemble_image1()
    print("Passed test: reassemble_image1()")

    test_reassemble_image2()
    print("Passed test: reassemble_image2()")

    test_reassemble_image3()
    print("Passed test: reassemble_image3()")

    test_cosine_similarity()
    print("Passed test: cos_sim()")

    print("Everything passed")
