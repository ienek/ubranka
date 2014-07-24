__author__ = 'ienek'

from skimage import io, filter, feature
from skimage.color import rgb2gray

from sklearn.feature_extraction import image
# test to load any image
test = io.imread('n02854926_28_0.jpg')

# loading to array
img_arr = image.img_to_graph(test)

patch_g = image.extract_patches(test, (4, 4, 2))

patch_g2 = image.extract_patches_2d(test, (4, 4))

print("Patch 1: %s " % str(patch_g.shape))
print("Patch 2: %s " % str(patch_g2.shape))

# grayscaling
gray_test = rgb2gray(test)

# building hog
hog_ = feature.hog(gray_test)

# building surf
# ORB: An efficient alternative to SIFT and SURF
surf_ = feature.ORB()
surf_.detect_and_extract(gray_test)

print("HOG: %s" % str(hog_.shape))
print("ORB: %s" % str(surf_.descriptors.shape))

# printing images
#test_ = filter.sobel(gray_test)
#io.imshow(gray_test)
#io.show()
