import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def get_img_total_for_digit(img, digit):
  dig_avg_img = dig_avg_imgs[:,:,digit].view()
  dig_avg_image0.shape = (784,1)
  dig_avg_image0.shape
  dig_avg_image0.transpose().dot(dig_avg_image0)
  simple_prod = []
  simple_prod_neg = []
  for row in range(MNH):
    for col in range(MNW):
      simple_prod.append(dig_avg_imgs[row,col,digit]*img[row,col])
      simple_prod_neg.append((1-dig_avg_imgs[row,col,digit])*img[row,col])
  return sum(simple_prod) - sum(simple_prod_neg)

def init_results():
  return dict((digit,0) for digit in range(10))

a = array([0, 1, 2, 3, 4, 5])
a

b = array([5, 6, 7, 8, 9, 10])
a.shape = (2,3)
b.shape = (2,3)
a.dot(b)
np.inner(a,b)
a[:,None]*b
c = a.flatten()
c[1] = 10
a
a.flatten().dot(b.flatten())
a.shape
a*b
0+6+14+24+36+50


dig_avg_imgs = np.load('../data/dig_avg_imgs.npy')

from load_mnist import get_mnist_sets

train_set, valid_set, test_set = get_mnist_sets()

len(train_set[1])
len(valid_set[1])
len(test_set[1])

MNH = 28
MNW = 28

dig_avg_imgs[14,14,:]
dig_avg_imgs[:,:,1].shape
dig_avg_imgs[:,:,1].shape

start_ind = 0 if len(sys.argv) <= 1 else int(sys.argv[1])
end_ind = len(test_set[1]) if len(sys.argv) <= 2 else int(sys.argv[2])

print "from","to",start_ind, end_ind

dig_counts = defaultdict(int) #counts of images for each digit
correct_counts = defaultdict(int) #counts of images for each digit
results = defaultdict(init_results)
total_correct = 0
samples = end_ind - start_ind
for i in range(start_ind, end_ind):
  img = test_set[0][i].reshape(28,28)
  actual_digit = test_set[1][i]
  all_vals = [get_img_total_for_digit(img, digit) for digit in range(10)]
  max_val, max_digit = max((v, i) for i, v in enumerate(all_vals))
  if(max_digit == actual_digit):
    correct_counts[max_digit] += 1
    total_correct += 1
  dig_counts[actual_digit] += 1
  results[actual_digit][max_digit] += 1


for digit in range(10):
  pcnt = float(correct_counts[digit])/dig_counts[digit] if dig_counts[digit] > 0 else 0
  print "\n{digit}: ({yes}/{total} is {pcnt})".format(digit=digit, yes=correct_counts[digit], total=dig_counts[digit], pcnt=pcnt)
  dig_results = sorted(results[digit], key=results[digit].get, reverse=True)
  for dig_guess in dig_results:
    print "{dig_guess}: {count}".format(dig_guess=dig_guess, count=results[digit][dig_guess])

print "total:",total_correct,"of",samples,"-",float(total_correct)/samples
