"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    #flip the kernel about both the axes
    ker = np.flipud(np.fliplr(kernel))
    
    #add a padding of zeroes around the image
    padding_y = int(Hk/2)
    padding_x = int(Wk/2)
    
    image_padded = np.zeros((Hi + padding_y*2, Wi + padding_x*2))
    image_padded[padding_y:-padding_y, padding_x:-padding_x] = image
    
    for m in range(Hi):
        for n in range(Wi):
            sumc = 0.0
            for i in range(Hk):
                for j in range(Wk):
                    sumc = sumc + ker[i,j]*image_padded[m+i,n+j]
                    #print("m=%d, n=%d, sumc=%d"%(m,n,sumc))
            out[m,n] = sumc

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    
    image_padded = np.zeros((H + pad_height*2, W + pad_width*2))
    image_padded[pad_height:-pad_height, pad_width:-pad_width] = image
    
    return image_padded


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #flip the kernel about both the axes
    ker = np.flipud(np.fliplr(kernel))
    
    #pad the image
    padded_img = zero_pad(image, Hk//2, Wk//2)
    
    for m in range(Hi):
        for n in range(Wi):
            out[m,n] = np.sum(ker*padded_img[m:m+Hk, n:n+Wk])
            
    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #flip the kernel about both the axes
    ker = np.flipud(np.fliplr(kernel))
    
    #pad the image
    padded_img = zero_pad(image, Hk//2, Wk//2)
    
    for m in range(Hi):
        for n in range(Wi):
            out[m,n] = np.sum(ker*padded_img[m:m+Hk, n:n+Wk])
            
    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    ker = np.flipud(np.fliplr(g))
    out = conv_fast(f, ker)
    
    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ker = np.flipud(np.fliplr(g))
    ker = ker - np.mean(ker)
    out = conv_fast(f, ker)
    
    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    g = (g-np.mean(g))/np.std(g)
    
    out = np.zeros((Hi, Wi))

    #pad the image
    padded_img = zero_pad(f, Hk//2, Wk//2)
    
    for m in range(Hi):
        for n in range(Wi):
            normalized_subimage = (padded_img[m:m+Hk, n:n+Wk] - np.mean(padded_img[m:m+Hk, n:n+Wk]))/np.std(padded_img[m:m+Hk, n:n+Wk])
            out[m,n] = np.sum(g*normalized_subimage)
            
    return out

    