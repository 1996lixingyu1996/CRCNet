import numpy as np

def IdenMatterLocation2(i, multiArg):
    """

    :param multiArg a list contains parameter: UNetImg, slidingLength, w0
    :param UNetImg: numpy array a gray img
    :param i: a number ith row
    :param slidingLength 10
    :param w0  int number whole slide width in dimension 0
    :return: a list which contains i, j  i:row information; j: column information
    """

    # a list that its elements(list) contains concrete information

    UNetImg = multiArg[0]
    slidingLength = multiArg[1]
    w0 = multiArg[2]
    wholePatch = multiArg[3]
    thumbPatch = multiArg[4]

    location = []
    ## this part can use jit to accelerate speed
    for j in range(int(w0 / (wholePatch * slidingLength)) + 1):

        patchImg = UNetImg[int(i * thumbPatch):int((i + 1) * thumbPatch),
                   int(j * thumbPatch * slidingLength):int((j + 1) * thumbPatch * slidingLength)]

        sum_10_patch = np.sum(patchImg)

        if sum_10_patch < thumbPatch * thumbPatch * slidingLength * 0.5 * 255:

            if j < int(w0 / (wholePatch * slidingLength)):

                temp = [[i, j]]
                location = location + temp

        # the last part must be returned
        if j == int(w0 / (wholePatch * slidingLength)):

            temp = [[i, j]]  # the last part, which needs detail calculation
            location = location + temp

    return location
