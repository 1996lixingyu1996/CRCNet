import numpy as np
from PIL import Image
from numba import jit
'''
NormMacenko: Normalise the appearance of a Source Image to a Target 
% image using Macenko's method.
% 
%
% Input:
% Img   - RGB Source image.  Numpy array
% Target   - RGB Reference image.
 # Relativative information are loaded by numpy file 
'/data/lxy/FirstProject/test/normalData/maxCRef.npy' MaxC
"/data/lxy/FirstProject/test/normalData/HERef.npy", HE
% Io       - (optional) Transmitted light intensity. (default 255)
% beta     - (optional) OD threshold for transparent pixels. (default 0.15)
% alpha    - (optional) Tolerance for the pseudo-min and pseudo-max. (default 1)                     

% Output:
% Norm     - Normalised RGB Source image.
%
%
% References:
% [1] M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley, X Guan, C 
%     Schmitt, NE Thomas. "A method for normalizing histology slides for 
%     quantitative analysis". IEEE International Symposium on Biomedical 
%     Imaging: From Nano to Macro, 2009 vol.9, pp.1107-1110, 2009.
%
%
% Acknowledgements:
%       This function is inspired by Mitko Veta's Stain Unmixing and Normalisation 
%       code, which is available for download at Amida's Website:
%       http://amida13.isi.uu.nl/?q=node/69
%       and Department of Computer Science,
%       University of Warwick, UK

'''
# @jit(nopython=True, parallel=True, nogil=True, cache=True)
def normal_Macenko(img, HERef, maxCRef, Io=255, alpha=1, beta=0.15):

    img = img.astype(np.float64)
    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float64) + 1) / Io)

    OD = OD.astype(np.float64)
    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # add a dimension according to the MATLAB code
    # by cross the H & E

    HE2 = np.cross(HE[:, 0], HE[:, 1])
    HE2 = HE2 / (HE2 ** 2).sum() ** 0.5
    HE = np.column_stack((HE, HE2))

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations

    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99), np.percentile(C[2, :], 99)])

    tmp = np.divide(maxC, maxCRef)

    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))

    Inorm[Inorm > 255] = 254

    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Inorm




