import cPickle, gzip
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

"""
usage:
python emf.py digit_of_interest [num_examples_to_show=30]
e.g.
python emf.py 6 50
"""


# Load the dataset
f = gzip.open('../data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set_size = len(train_set[1])
valid_set_size = len(valid_set[1])
test_set_size = len(test_set[1])

start_ind = 0 if len(sys.argv) <= 1 else int(sys.argv[1])
end_ind = train_set_size if len(sys.argv) <= 2 else int(sys.argv[2])

MNH = 28 #height of mnist images
MNW = 28 #width of mnist images

dig_avg_imgs = np.zeros((MNH,MNW,10,)) #average pixel val for each digit
dig_counts = defaultdict(int) #counts of images for each digit
for i in range(start_ind, end_ind):
  digit = train_set[1][i]
  img = train_set[0][i].reshape(MNH,MNW)
  for (row,col),pxval in np.ndenumerate(img):
    dig_avg_imgs[row,col,digit] += pxval
  dig_img_num = dig_counts[digit]
  dig_counts[digit] += 1

for (row,col,digit),pxval_total in np.ndenumerate(dig_avg_imgs):
  num_images = float(dig_counts[digit])
  dig_avg_imgs[row,col,digit] = pxval_total/num_images

for digit in range(10):
  ave_img = dig_avg_imgs[:,:,digit]
  print digit, ":", dig_counts[digit]
  plt.imshow(ave_img, cmap=plt.cm.gray)
  plt.show()

fpath = '../data/dig_avg_imgs'
np.save(fpath, dig_avg_imgs)

# for each distro:
  # produce a distribution for each pixel so you can compute
  # P(pxval|digit) - prob of a particular value at the pixel, given that the digit is X
  # and P(pxval) - prob of that particular value at the pixel regardless of the digit
  #                being represented
  # intuitively what this will give you is certain pixels that are discriminative
  # for particular digits ie. where P(0.9|"7") at a certain pixel is high
  # but P(0.9 | "1"-"6","8","9") at that pixel is low overall...
  #
  # then P(digit|pxval) = P(pxv|dig)*P(dig)/P(pxv)
  # so you'd be able to compute both an expected value for the pixel (ev = average over all images for each digit)
  # and also a percentage of images where that pixel had a value (fa). Since colour is reasonably constant (c),
  # one would expect fa*c ~= ev
  #
  # To get shape, what about trying to localise the pixel in the shape:
  #   get connected horizontal lines
  #   get connected vertical lines
  #   their width, start and end points would have an interesting distribution

# this system is sensitive to scaling, translations and rotations. it's essentially
# building a probabilistic template to match against. To normalise against translations
# could try a bounding box and pixel distros are done as offsets from the corners of the bounding box.
# For rotations potentially interest points or shape matching could be used to try determine orientation,
# but a simple scheme might try various rotations of the input image in parallel.
# Scaling might be handled by resizing based on bounding box
#
# n.b. non-parametric density estimation


def fit_bbox(img):
  pass

