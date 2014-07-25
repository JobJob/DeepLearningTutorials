import cPickle, gzip
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
from jutils import makedirs
from skimagej import save_image

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

print len(train_set[1])
print len(valid_set[1])
print len(test_set[1])

start_ind = 0 if len(sys.argv) <= 1 else int(sys.argv[1])
end_ind = 10 if len(sys.argv) <= 2 else int(sys.argv[2])

dig_counts = defaultdict(int) #counts of images for each digit
for i in range(start_ind, end_ind):
  img = train_set[0][i].reshape(28,28)
  #plt.imshow(img, cmap=plt.cm.gray)
  #plt.show()
  digit = train_set[1][i]
  dig_img_num = dig_counts[digit]
  dig_counts[digit] += 1
  fpath = "../data/mnist/train/images of {0}s/{0}_{1}.jpg".format(digit, dig_img_num)
  if(dig_img_num == 0):
    makedirs(fpath)
  save_image(fpath, img)

