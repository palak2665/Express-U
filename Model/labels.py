import os
import os.path
from pathlib import Path
import glob
import numpy as np
import pandas as pd


def labels_generator(path,name):

    classes = os.listdir(path)
    filepaths = []
    labels = []
    for d in classes:
        filename = os.listdir(path + d)
        for f in filename:
            fpath = os.path.join(path + d + '/' + f)
            filepaths.append(fpath)
            labels.append(d)

    fileSeries = pd.Series(filepaths,name ='file_paths')
    labelseries = pd.Series(labels,name ='labels')
    df = pd.concat([fileSeries,labelseries],axis =1)
    df = pd.DataFrame(np.array(df),columns=['file_paths','labels'])
    df.to_csv(f'./data/preprocessed/{name}.csv',index=False)

labels_generator('E:/Documents/GitHub/Express-U/data/preprocessed/canny/','canny_labels')
labels_generator('E:/Documents/GitHub/Express-U/data/preprocessed/laplacian/','laplacian_labels')