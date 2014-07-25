import cPickle, gzip
import numpy as np
import scipy as sp
import theano
import theano.tensor as T
from PIL import Image


# Load the dataset
f = gzip.open('../data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
len(train_set[1])
len(valid_set[1])
len(test_set[1])

def shared_dataset(data_xy):
  """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
  data_x, data_y = data_xy
  shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
  shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
  # When storing data on the GPU it has to be stored as floats
  # therefore we will store the labels as ``floatX`` as well
  # (``shared_y`` does exactly that). But during our computations
  # we need them as ints (we use labels as index, and if they are
  # floats it doesn't make sense) therefore instead of returning
  # ``shared_y`` we will have to cast it to int. This little hack
  # lets us get around this issue
  return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500    # size of the minibatch

# accessing the third minibatch of the training set

data  = train_set_x[2 * batch_size: 3 * batch_size]
label = train_set_y[2 * batch_size: 3 * batch_size]


imgname = "bananas"
imd = test_set[0][1].reshape(28,28)
sp.misc.imsave(imgname+".png", imd)

filter_names = ["sobel","prewitt","laplace"]
filters = {filter_name: getattr(sp.ndimage.filters, filter_name) for filter_name in filter_names}
imagefns = []
for fltrname,fltr in filters.iteritems():
  imgnamefl = imgname+"_"+fltrname+".png"
  sp.misc.imsave(imgnamefl, fltr(imd))
  imagefns.append(imgnamefl)

import os, sys
# os.system("open {0}".format(" ".join(imagefns)))

from skimage import data, io, filter

image = data.coins() # or any NumPy array!
edges = filter.sobel(imd)
#io.imshow(edges)
# io.show()
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from matplotlib import pyplot as plt
# coords = corner_peaks(corner_harris(imd), min_distance=5)
# coords
# coords_subpix = corner_subpix(imd, coords, window_size=13)
# fig, ax = plt.subplots()
# ax.imshow(imd, interpolation='nearest', cmap=plt.cm.gray)
# ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
# ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
# ax.axis((0, 28, 28, 0))
# plt.show()

plt.figure()

from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img1 = rgb2gray(data.lena())
img1.shape
imd.shape
img1.dtype
imd.dtype
imd = np.asfarray(imd)
img1
imd
descriptor_extractor = ORB()

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

keypoints1
descriptors1
plt.imshow(imd, cmap=plt.cm.gray)
plt.show()
plt.imshow(img1, cmap=plt.cm.gray)
# print "keypoints1",keypoints1
# print "descriptors1",descriptors1
exit()


from skimage import measure

contour_counts = {0:2, 1:1, 2:2, 3:1, 4:1, 5:1, 6:2, 7:1, 8:3, 9:2 }

wrongs = []
THRESH = float(sys.argv[1])

for i in range(100):
  imd = test_set[0][i]
  digit = test_set[1][i]

  non_zeros = imd > THRESH
  imd[non_zeros] = 1.0
  imd = imd.reshape(28,28)

  # Find contours at a constant value of 0.8
  contours = measure.find_contours(imd, 0.2)
  num_contours = len(contours)
  if num_contours > contour_counts[digit]:
    fig, ax = plt.subplots()
    ax.imshow(imd, interpolation='nearest', cmap=plt.cm.gray)
    print i,":",digit,"-",num_contours
    wrongs.append(i)
    for n, contour in enumerate(contours):
      # plot all contours found
      ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    #Display the image with the contours
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

print(wrongs)
