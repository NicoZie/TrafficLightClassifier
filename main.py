import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

%matplotlib inline

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"
# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label
image_num = 0
selected_image = IMAGE_LIST[image_num][0]
selected_label = IMAGE_LIST[image_num][1]

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
plt.imshow(selected_image)
print("Shape: "+str(selected_image.shape))
print("Label (red, yellow, green): " + str(selected_label))
# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    row_crop = 2
    col_crop = 8
    standard_im = image[row_crop:-row_crop, col_crop:-col_crop, :]
    standard_im = cv2.resize(standard_im, (32, 32))
    
    return standard_im
plt.imshow(standardize_input(selected_image))
## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = []
    if label == "red":
        one_hot_encoded = [1, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0]
    else:
        one_hot_encoded = [0, 0, 1]
    
    return one_hot_encoded

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

## TODO: Display a standardized image and its label
image_num = 1110
standard_image = STANDARDIZED_LIST[image_num][0]
standard_label = STANDARDIZED_LIST[image_num][1]

plt.imshow(standard_image)
print("Shape: "+str(standard_image.shape))
print("Label (red, yellow, green): " + str(standard_label))

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image):
    
    ## TODO: Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # HSV channels
    #h = hsv[:,:,0]
    #s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    ## TODO: Create and return a feature value and/or vector
    
    #feature: area of brightness (v-value)
    # 1 step: add all the v-values in a row and create a list with len(#rows)
    row_values = np.sum(v, axis = 1)
    list_rv = row_values.tolist()
    feature = []
    
    #2 step: define a group of rows, where red, yellow, green is expected and sum_up the rows of the v-values
    red_area = sum(list_rv[2:10 + 1])
    feature.append(red_area)
    yellow_area = sum(list_rv[11:20 + 1])
    feature.append(yellow_area)
    green_area = sum(list_rv[22:30 + 1])
    feature.append(green_area)

    #plot_bar(row_values)
    #return the sumed-upp v-values for red, yellow and green area
    return feature

#Helping-function to plot a bar-chart with the row values
def plot_bar(values):
    plt.barh(np.arange(len(values)), values, align='center', alpha=0.5)
    plt.yticks(np.arange(len(values)), values)
    
#print(create_feature(standard_image)) 

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    
    # label the area with the highest value
    predicted_label = []
    v_area = create_feature(rgb_image)
    if v_area[0] > v_area[1] and v_area[0] > v_area[2]:
        predicted_label = [1, 0, 0]
    elif v_area[1] > v_area[0] and v_area[1] > v_area[2]:
        predicted_label = [0, 1, 0]
    else:
        predicted_label = [0, 0, 1]

    return predicted_label 

    # Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
num = 10
test_mis_im = MISCLASSIFIED[num][0]
plt.imshow(test_mis_im)
print(MISCLASSIFIED[num][1])
print(create_feature(test_mis_im))

# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")