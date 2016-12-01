from __future__ import print_function

import numpy as np
import tflearn

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('data.csv', target_column=13, categorical_labels=True, n_classes=10)

print('Load data finished')

index = int(len(data)*0.8);

training_data = data[0:index]; 
training_label = labels[0:index];

test_data = data[index:];
test_label = labels[index:];

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
print('start training')
# Start training (apply gradient descent algorithm)
model.fit(training_data, training_label, n_epoch=100, batch_size=100, show_metric=True)

print('predict')
pred = model.predict(test_data)

count = float(0);
for idx, val in enumerate(pred):
    label = test_label[idx];
    if np.argmax(label) == np.argmax(val):
        count += 1;
accuracy=count/len(test_label);
print(accuracy);