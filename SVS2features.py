import os
import glob
import cv2 as cv
import pandas as pd
import openslide
import numpy as np
import concurrent.futures
import tensorflow as tf

from IdenMatterLocation import *
from ImgDetail import *

from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.xception import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


if __name__ == '__main__':

    slidingLength = 10
    poolNum = 80
    WSI_csv_path = ""
    SVSfolderPath = ''
    thumbs_paths = ''  

    Ref = cv.imread('')
    Ref = cv.resize(Ref, (224, 224))

    feature_tensor_save_path = ""
    probability_tensor_save_path = ""

    if not os.path.exists(feature_tensor_save_path):
        os.mkdir(feature_tensor_save_path)
    if not os.path.exists(probability_tensor_save_path):
        os.mkdir(probability_tensor_save_path)

    Ref1 = np.expand_dims(Ref, axis=0)

    probability_model = tf.keras.models.load_model("")
    xception_model = tf.keras.Model(inputs=probability_model.input, \
                                outputs=probability_model.outputs + [probability_model.get_layer("Dense256").output])
    data = pd.read_csv(WSI_csv_path)

    paths = [data['png_paths'].iloc[i] for i in range(len(data['png_paths']))]

    for path in paths:
        midname = path[path.rindex("/") + 1:path.rindex(".")] 

        xception_feature_save_path = feature_tensor_save_path + "/" + midname + ".npy"
        xception_probability_save_path = probability_tensor_save_path + "/" + midname + ".npy"

        try:
            xception_feature_path_mark = glob.glob(xception_feature_save_path)[0]
            if xception_feature_path_mark:

                continue

        except IndexError:

            pass

        SVSpath = data[data["ParentSpecimen"] == midname]["svs_paths"].iloc[0]
        print("SVSPath:", SVSpath)
        slide = openslide.open_slide(SVSpath)

        [w0, h0] = slide.level_dimensions[0]

        # some slides may don't have dimension 3
        try:

            [w3, h3] = slide.level_dimensions[3]

        except IndexError:

            [w1, h1] = slide.level_dimensions[1]
            w3 = int(w1 / 16)
            h3 = int(h1 / 16)

        slide.close()

        # UNetImg adjustment
        unet_img_path = "/data/lxy/MCO_thumb/MCO_thumb/" + midname + ".png"
        UNetImg = cv.imread(unet_img_path, 0)
        UNetImg = cv.resize(UNetImg, (w3, h3))
        # print("MPP:", data[data["ParentSpecimen"] == midname]["MPP"].iloc[0])
        SVSmpp = float(data[data["ParentSpecimen"] == midname]["MPP"].iloc[0])

        if np.isnan(SVSmpp):
            continue

        wholePatch = int(round(224 * 0.5 / SVSmpp))  # get the patch with 0.5 um/mpp
        thumbPatch = int(round(wholePatch * w3 / w0))  # patch length of thumb

        xception_feature_mat = np.zeros((int(h0 / wholePatch), int(w0 / wholePatch), 256), dtype=np.float32)
        xception_probability_mat = np.zeros((int(h0 / wholePatch), int(w0 / wholePatch), 9), dtype=np.float32)

        with concurrent.futures.ProcessPoolExecutor() as executor:

            rows = range(int(h0 / wholePatch))
            multiArg = [UNetImg, slidingLength, w0, wholePatch, thumbPatch]
            matterLocation = executor.map(partial(IdenMatterLocation2, multiArg=multiArg), rows)

        roughLocation = []

        for i in matterLocation:

            roughLocation = roughLocation + i

        # print(roughLocation)
        pool = Pool(poolNum)

        countFlag = 0
        full = False 
        q = Manager().Queue(300)

        while(True):

            multiArg = [roughLocation[countFlag], SVSmpp, w3, h3, w0,
                    path, SVSpath]  # roughLocation[i] i, j(row, column information)

            pool.apply_async(strip2Img, args=(multiArg, q,))

            if q.empty():
                xception_model.predict(Ref1)
            if not q.empty():
                
                location = q.get_nowait()  
                length = len(location)

                if length > 0:

                    for flag in range(length):

                        img = location[flag][0]
                        i = location[flag][1]
                        j = location[flag][2]
                        k = location[flag][3]

                        x = np.expand_dims(img, axis=0)
                        x1 = preprocess_input(x)

                        probability_feature, xception_feature = xception_model.predict(x1)
                        xception_feature_mat[i, j * slidingLength + k, :] = xception_feature  
                        xception_probability_mat[i, j * slidingLength + k, :] = probability_feature  

            countFlag += 1

            if countFlag == len(roughLocation):
                break

        while(True):

            if not q.empty():

                location = q.get_nowait()

                length = len(location)

                if length >= 0:

                    for flag in range(length):

                        img = location[flag][0]
                        i = location[flag][1]
                        j = location[flag][2]
                        k = location[flag][3]

                        x = np.expand_dims(img, axis=0)
                        x1 = preprocess_input(x)
                        probability_feature, xception_feature = xception_model.predict(x1)
                        xception_feature_mat[i, j * slidingLength + k, :] = xception_feature  # xception_model.predict(x1)
                        xception_probability_mat[i, j * slidingLength + k, :] = probability_feature  # probability_model.predict(x1)
            if q.empty():

                break

        pool.close()
        pool.join()

        np.save(xception_feature_save_path, xception_feature_mat)
        np.save(xception_probability_save_path, xception_probability_mat)