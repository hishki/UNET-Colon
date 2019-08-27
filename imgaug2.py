import os
import random
from scipy import ndarray

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

folder_path = '/home/vedula/data/train/Annotation'
num_files_desired = len(os.listdir(folder_path))    
# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in enumerate(sorted(os.listdir(folder_path))) if os.path.isfile(os.path.join(folder_path, f))]
print(images)

num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    print(num_generated_files)
    image_path = images[num_generated_files]
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    
    #num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 1
    transformed_image = None
    while num_transformations <= 3:
        # random transformation to apply for a single image
        # key = random.choice(list(available_transformations))
        transformed_image = available_transformations['rotate'](image_to_transform)
        num_transformations += 1
        aug_path = '/home/vedula/data/train/augmented_annotations'
        new_file_path = '%s/augmented_image_%s.jpg' % (aug_path, num_generated_files)
        # write image to the disk
        io.imsave(new_file_path, transformed_image)

        transformed_image = available_transformations['noise'](image_to_transform)
        num_transformations += 1
        aug_path = '/home/vedula/data/train/augmented_annotations'
        new_file_path = '%s/augmented_image_%s.jpg' % (aug_path, num_generated_files)
        # write image to the disk
        io.imsave(new_file_path, transformed_image)

        transformed_image = available_transformations['horizontal_flip'](image_to_transform)
        num_transformations += 1
        aug_path = '/home/vedula/data/train/augmented_annotations'
        new_file_path = '%s/augmented_image_%s.jpg' % (aug_path, num_generated_files)
        # write image to the disk
        io.imsave(new_file_path, transformed_image)
    num_generated_files += 1