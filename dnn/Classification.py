import numpy as np
import tflearn
import matlab.engine
import operator
import sys

# To use matlab.engine, please go to /Applications/MATLAB_R2016a.app/extern/engines/python/
# and run python setup.py install
# https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

def main():

    filename = sys.argv[1];
    engine = matlab.engine.start_matlab();
    data = engine.featureExtractionForSingleFile(filename);

    print('done');

    print('Load data finished')

    print('building neural network');
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

    print('start loading model')
    model.load('./model/dnn_model', False);

    print('predict...')
    pred = model.predict(data)

    type_to_class = {
        0:'blues',
        1:'classical',
        2:'country',
        3:'disco',
        4:'hiphop',
        5:'jazz',
        6:'metal',
        7:'pop',
        8:'reggae',
        9:'rock'
    };

    dict = {};
    for idx, val in enumerate(pred):
        type = np.argmax(val);
        dict[type] = dict.get(type, 0) + 1;
    print(dict)
    print(type_to_class)
    print(type_to_class.get(max(dict.items(), key=operator.itemgetter(1))[0], 'no class found'));

if __name__ == "__main__":
    main()