import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
from glob import glob
import os
import imageio
import numpy as np


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, real_images, fake_images):
    # calculate activations
    act1 = model.predict(real_images)
    act2 = model.predict(fake_images)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
def get_images(filename):
    x = imageio.imread(filename)
    return x


# load cifar10 images
# filenames = glob(os.path.join('./target_deer', '*.*'))
# real_images = [get_images(filename) for filename in filenames]
# real_images = np.array(real_images)
#
# filenames = glob(os.path.join('./fake_dear', '*.*'))
# fake_images = [get_images(filename) for filename in filenames]
# fake_images = np.array(fake_images)

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# load cifar10 images
filenames = glob(os.path.join('./Target_cifar10', '*.*'))
real_images = [get_images(filename) for filename in filenames]
real_images = np.array(real_images)

filenames = glob(os.path.join('D:/test11111111111111/Self-Attention-GAN-Tensorflow-master\Self-Attention-GAN-Tensorflow-master/results/SAGAN_cifar10_hinge_32_128_True', '*.*'))
fake_images = [get_images(filename) for filename in filenames]
fake_images = np.array(fake_images)
print('Loaded', real_images.shape, fake_images.shape)
# convert integer to floating point values
real_images = real_images.astype('float32')
fake_images = fake_images.astype('float32')
# resize images
real_images = scale_images(real_images, (299,299,3))
fake_images = scale_images(fake_images, (299,299,3))
print('Scaled', real_images.shape, fake_images.shape)
# pre-process images
real_images = preprocess_input(real_images)
fake_images = preprocess_input(fake_images)
# calculate fid
fid = calculate_fid(model, real_images, fake_images)
print('FID: %.3f' % fid)
