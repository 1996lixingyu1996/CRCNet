import openslide
import numpy as np
import glob
import cv2

from NormMacenko import *
#
'''
This code is used to test server effiency.
'''

def strip2Img(multiArg, q):
    '''

    :param multiArg: A parameter list which contains UNetImg,
    :param UNetImg
    :param i int location information
    :param j int location information
    :return: a list which contains transfered imgs(its length smaller than 10)
    '''

    i = multiArg[0][0]
    j = multiArg[0][1]
    SVSmpp = float(multiArg[1])
    w0 = multiArg[4]
    SVSpath = multiArg[6]  # SVS file path

    result = []  # return list

    slidingLength = 10

    HERef = np.load("/data/lxy/pycharm_project/HERef.npy")
    maxCRef = np.load('/data/lxy/pycharm_project/maxCRef.npy')

    slide = openslide.open_slide(SVSpath)

    wholePatch = int(round(224 * 0.5 / SVSmpp))  # get the patch with 0.5 um/mpp,  must int data type
    max_threshold = wholePatch * wholePatch * 0.9 * 255
    min_threshold = wholePatch * wholePatch * 0.2

    if j < int(w0 / (wholePatch * slidingLength)):

        for k in range(10):

            slidePatch = slide.read_region(
                (j * slidingLength * wholePatch + k * wholePatch, i * wholePatch), 0,
                (wholePatch, wholePatch))
            # print("slidePatch:", slidePatch)
            np_slidePatch = np.array(slidePatch)
            # print("np_slidepatch:", np_slidePatch.shape)
            np_slidePatch = cv2.cvtColor(np_slidePatch, cv2.COLOR_RGBA2RGB)
            np_slidePatch_gray = cv2.cvtColor(np_slidePatch, cv2.COLOR_RGB2GRAY)
            sum_single_patch = np.sum(np_slidePatch_gray)
            np_slidePatch = cv2.resize(np_slidePatch, (224, 224))
            # print("Strip2Imags")
            if sum_single_patch < max_threshold and sum_single_patch > min_threshold:

                try:

                    img_Mac = normal_Macenko(np_slidePatch, HERef, maxCRef, Io=255, alpha=1, beta=0.15)
                    temp = [[img_Mac, i, j, k]]
                    result = result + temp

                except np.linalg.LinAlgError:

                    pass

    if j == int(w0 / (wholePatch * slidingLength)):

        remained = int(w0 / wholePatch) - int(w0 / (wholePatch * slidingLength)) * slidingLength

        for k in range(remained):

            slidePatch = slide.read_region(
                (j * slidingLength * wholePatch + k * wholePatch, i * wholePatch), 0,
                (wholePatch, wholePatch))
            np_slidePatch = np.array(slidePatch)
            np_slidePatch = cv2.cvtColor(np_slidePatch, cv2.COLOR_RGBA2RGB)
            np_slidePatch_gray = cv2.cvtColor(np_slidePatch, cv2.COLOR_RGB2GRAY)
            sum_single_patch = np.sum(np_slidePatch_gray)
            np_slidePatch = cv2.resize(np_slidePatch, (224, 224))

            if sum_single_patch < max_threshold and sum_single_patch > min_threshold:  # wholePatch -> a parameter,only calculate once

                try:

                    img_Mac = normal_Macenko(np_slidePatch, HERef, maxCRef, Io=255, alpha=1, beta=0.15)

                    temp = [[img_Mac, i, j, k]]
                    result = result + temp

                except np.linalg.LinAlgError:

                    pass

    q.put(result)

    slide.close()
    return result
