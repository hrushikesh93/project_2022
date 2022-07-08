# Imports
import os
import numpy as np
from scipy.linalg import svd

"""
Singular Value Decomposition
"""
# define a matrix
X = np.array([[3, 3, 2], [2,3,-2]])
print(X)
# perform SVD
U, singular, V_transpose = svd(X)
# print different components
print("U: ",U)
print("Singular array",singular)
print("V^{T}",V_transpose)

"""
Calculate Pseudo inverse
"""
# inverse of singular matrix is just the reciprocal of each element
singular_inv = 1.0 / singular
# create m x n matrix of zeroes and put singular values in it

s_inv = np.zeros(X.shape)
s_inv[0][0]= singular_inv[0]
s_inv[1][1] =singular_inv[1]
# calculate pseudoinverse
M = np.dot(np.dot(V_transpose.T,s_inv.T),U.T)
print(M)
print("\n\nAGE: 37")
os.system("figlet 37")
print("\n\nGENDER: Male")
os.system("figlet Male")
"""
SVD on image compression
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data
from skimage.color import rgb2gray

cat = skimage.io.imread('/home/kali/Downloads/fingerimage.jpg')
plt.imshow(cat)
# convert to grayscale
gray_cat = rgb2gray(cat)

# calculate the SVD and plot the image
U,S,V_T = svd(gray_cat, full_matrices=False)
S = np.diag(S)
fig, ax = plt.subplots(5, 2, figsize=(8, 20))

curr_fig=0
for r in [5, 10, 70, 100, 200]:
    cat_approx=U[:, :r] @ S[0:r, :r] @ V_T[:r, :]
    ax[curr_fig][0].imshow(256-cat_approx)
    ax[curr_fig][0].set_title("k = "+str(r))
    ax[curr_fig,0].axis('off')
    ax[curr_fig][1].set_title("Original Image")
    ax[curr_fig][1].imshow(gray_cat)
    ax[curr_fig,1].axis('off')
    curr_fig +=1
plt.show()
