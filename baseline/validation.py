from __future__ import print_function

import tflearn

def build_load_model():
    print('Building neural network...')
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
    model.load('model_{}/dnn_model'.format(30))
    return model

def main():
    model = build_load_model()

if __name__ == "__main__":
    main()

