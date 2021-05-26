from skimage.io import imread
import matplotlib.pyplot as plt
import skimage.color as color 
import skimage.filters as filters
from skimage.util import random_noise
from scipy import ndimage
from sklearn import cluster
from skimage.segmentation import slic
import skimage.transform as transform
import skimage.feature as feature
import time

start = time.time()

"""1. Determine the size of the avengers imdb.jpg image. Produce a grayscale and a black-
and-white representation of it."""

#read the image
avengers_original = imread('data/avengers_imdb.jpg')
#find the size of the image and display it
sizeofimage = avengers_original.size
print("Size of the 'avengers_imdb.jpg' : ",sizeofimage)

#Convert the image object from RGB format to greyscale.
avengers_gray = imread('data/avengers_imdb.jpg', as_gray=True)
#save the image in outputs folder
plt.imsave('outputs/avengers_imdb_grayscale.jpg', avengers_gray, cmap = plt.cm.gray)

#Convert the image from Grayscale to Black and White version, using Otsu threshold
threshold = filters.threshold_otsu(avengers_gray)
avengers_bw = avengers_gray > threshold
#save the image in outputs folder
plt.imsave('outputs/avengers_imdb_blackwhite.jpg', avengers_bw,  cmap=plt.cm.gray)

#Display: original 'avengers_imdb' image and it's Grayscale & Black and White representation
fig, ((ax0,ax1, ax2)) = plt.subplots( nrows=1, ncols=3,  sharex=True, sharey=True )

ax0.imshow(avengers_original)
ax0.axis( 'off' )
ax0.set_title('Original')

#The resulting image is in grayscale. However, imshow, by default, uses a kind of heatmap to display the image intensities. 
#Hence, specifying the grayscale colormap as cmap=plt.cm.gray
ax1.imshow(avengers_gray, cmap=plt.cm.gray)
ax1.axis( 'off' )
ax1.set_title('Grayscale')

ax2.imshow(avengers_bw, cmap=plt.cm.gray)
ax2.axis( 'off' )
ax2.set_title('Black and White')

fig.tight_layout()
plt.show()


"""2. Add Gaussian random noise in bush house wikipedia.jpg (with variance 0.1) and filter
the perturbed image with a Gaussian mask (sigma equal to 1) and a uniform smoothing mask
(the latter of size 9x9). """

#read the image
bush_house_original = imread("data/bush_house_wikipedia.jpg")

#Add Gaussian random noise in bush_house_wikipedia.jpg (with variance 0.1)
gaussian_noise = random_noise(bush_house_original, mode='gaussian', var=0.1)
#save the image in outputs folder
plt.imsave('outputs/bush_house_gaussian_noise.jpg', gaussian_noise)

#filter the perturbed image with a Gaussian mask (sigma equal to 1)
gaussian_mask = filters.gaussian(gaussian_noise,  sigma=1, multichannel=False)
#save the image in outputs folder
plt.imsave('outputs/bush_house_gaussian_mask.jpg', gaussian_mask)

#uniform smoothing mask of size 9x9
uniform_mask = ndimage.uniform_filter(gaussian_noise, size=(9,9,1))
#save the image in outputs folder
plt.imsave('outputs/bush_house_uniform_mask.jpg', uniform_mask)

#display: 'bush_house_wikipedia' original image, gaussian random noise, gaussian mask and uniform smoothing mask
fig_bush_house, ((ax3, ax4, ax5, ax6)) = plt.subplots( nrows=1, ncols=4,  figsize=(15, 10), sharex=True, 
                                                        sharey=True )

ax3.imshow(bush_house_original)
ax3.axis( 'off' )
ax3.set_title('Original')

ax4.imshow(gaussian_noise)
ax4.axis( 'off' )
ax4.set_title('Gaussian noise')

ax5.imshow(gaussian_mask)
ax5.axis( 'off' )
ax5.set_title('Gaussian mask')

ax6.imshow(uniform_mask)
ax6.axis( 'off' )
ax6.set_title('Uniform smoothing mask')

plt.show()


""" 3. Divide forestry commission gov uk.jpg into 5 segments using k-means segmentation."""

#read the image
img_forestry = imread("data/forestry_commission_gov_uk.jpg") 

#create random noise
img_forest_noise= random_noise(img_forestry,mode='gaussian')
#add filter
img_forest_g_mask = filters.gaussian(img_forest_noise,sigma=2, multichannel=False)
#use slic function for image segmentation using k-means
segments = slic(img_forest_g_mask, compactness=12, n_segments=5, start_label =1)
plt.imsave('outputs/forestry_commission_gov_uk_kmeans.jpg', segments)

#display: 'forestry_commission_gov_uk' original image and k-means segmentation
fig, ((ax7,ax8)) = plt.subplots( nrows=1, ncols=2, sharex=True, sharey=True )

ax7.imshow(img_forestry)
ax7.axis( 'off' )
ax7.set_title('Original')

ax8.imshow(segments, interpolation='nearest')
ax8.axis( 'off' )
ax8.set_title('K-means segmentation')

fig.tight_layout()
plt.show()



#read image
img_rolland = imread("data/rolland_garros_tv5monde.jpg")
#convert the image to greyscale for canny edge detection
img_rolland_gray = color.rgb2gray(img_rolland)
#detect edges using Canny algorithm for 1 value of sigma
edges = feature.canny( img_rolland_gray, sigma=1 ) 
#apply classic straight-line Hough transform
line_hough = transform.probabilistic_hough_line( edges, threshold=10, line_length=5, line_gap=3 )

#display results
fig, ((ax9)) = plt.subplots( nrows=1, ncols=1, sharex=True, sharey=True )

for line_ in line_hough:
    p0, p1 = line_
    ax9.plot(( p0[0], p1[0] ), ( p0[1], p1[1] ))
ax9.set_xlim(( 0, img_rolland_gray.shape[1] ))
ax9.set_ylim(( img_rolland_gray.shape[0], 0 ))
ax9.axis( 'off' )
fig.tight_layout()

#save image in outputs folder
plt.savefig('outputs/Hough_Transform.png')
ax9.set_title( 'Canny edge detection and Hough transform' )
plt.show()


# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")