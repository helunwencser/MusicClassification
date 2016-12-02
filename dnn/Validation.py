import numpy as np
import tflearn
import matlab.engine
import operator
import os
import re


# To use matlab.engine, please go to /Applications/MATLAB_R2016a.app/extern/engines/python/
# and run python setup.py install
# https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

type_to_class = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae',
    9: 'rock'
}

types = {'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'}

def featureExtraction(filename):
    '''
    Get the feature of audio file
    :param filename: audio file name
    :return: audio feature
    '''
    engine = matlab.engine.start_matlab()
    return engine.featureExtractionForSingleFile(filename)

def load_model():
    '''
    Load model
    :return: model
    '''
    # Build neural network
    net = tflearn.input_data(shape=[None, 13])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net)
    # Define model
    model = tflearn.DNN(net)
    model.load('./model/dnn_model', False)
    return model

def classify(model, file):
    '''
    Classify audio file
    :param model: neural model
    :param file: audio file
    :return: audio file class
    '''
    pred = model.predict(featureExtraction(file))
    dict = {};
    for idx, val in enumerate(pred):
        type = np.argmax(val);
        dict[type] = dict.get(type, 0) + 1;
    return type_to_class.get(max(dict.items(), key=operator.itemgetter(1))[0], 'no class found')

def main():

    model = load_model();

    correct_result = 0

    directories = os.listdir('./data')
    for directory in directories:
        if directory in types:
            print('Validating type: {}'.format(directory))
            files = os.listdir('./data/{}'.format(directory))
            for file in files:
                if file.endswith('.au'):
                    index = int(re.findall('\d+', file)[0])
                    if index >= 80:
                        type = classify(model, './data/{0}/{1}'.format(directory, file))
                        print('\tValidating audio file: {0}/{1}, {2}'.format(directory, file, type))
                        if type == directory:
                            correct_result += 1
                            print('Correct: {}'.format(correct_result))

    print('{0} files out of {1} are correctly classified.'.format(correct_result, 200))
    print('The accuracy is {}.'.format(correct_result / float(200)))

if __name__ == "__main__":
    main()
