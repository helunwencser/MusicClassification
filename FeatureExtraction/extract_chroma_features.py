# https://github.com/tyiannak/pyAudioAnalysis/wiki

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import pandas as pd
import os

def extract_one(filename, folder):
    '''

    :param filename: path for the wav file
    :param folder: output folder path of the csv file
    :return:
    '''
    [Fs, x] = audioBasicIO.readAudioFile(filename)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.025*Fs, 0.010*Fs)
    filename = filename.split('/')[-1]
    output = os.path.join(folder, filename[:-3] + 'csv')
    df = pd.DataFrame(data=F.T.astype(float))
    if not os.path.exists(folder):
        os.makedirs(folder)
    df.to_csv(output, sep=',', header=False, float_format='%.5f', index=False)


def extract_allfile():
    '''
    extract 34 features for each audios for 10 genres and output csv file in the same folder structure feature.
    :return:
    '''
    root_folder = '../wav/genres'
    categories = os.listdir(root_folder)
    output_root = '../features'
    print categories
    for cat in categories:
        subfolder = os.path.join(root_folder,cat)
        output_sub = os.path.join(output_root,cat)
        for file in os.listdir(subfolder):
            extract_one(os.path.join(subfolder, file),output_sub)

        print cat

extract_allfile()