import numpy as np
import cv2 as cv
import csv
import os

class CNN:
    def __init__(self):
        w1, b1 = self.initializeParameters((3,3,3,5))
        w2, b2 = self.initializeParameters((3,3,5,10))
        w3, b3 = self.initializeParameters((23760,300))
        w4, b4 = self.initializeParameters((300,11))

        self.params = {
            "w1" : w1,
            "b1" : b1,
            "w2" : w2,
            "b2" : b2,
            "w3" : w3,
            "b3" : b3,
            "w4" : w4,
            "b4" : b4
        }

    def train(self, data, labels):

        for epoch in range(30):
            print('epoch:', epoch + 1)
            self.miniBatchGradientDescent(10, self.backwardPropagation, data, labels, self.params, self.forwardPropagation, 5)

    def predict(self):
        pass

    def initializeParameters(self, size):
        weights = np.random.normal(loc=0, scale=1, size=size)
        bias = np.zeros(size[-1])

        return weights, bias

    def saveParameters(self):
        pass

    def forwardPropagation(self, data, labels):
        c = self.convolutionalLayer(data, num_filters=5, parameters=(self.params["w1"], self.params["b1"]))
        #print(c.shape)
        p = self.maxPoolingLayer(c)
        #print(p.shape)
        c2 = self.convolutionalLayer(p, num_filters=10, parameters=(self.params["w2"], self.params["b2"]))
        #print(c2.shape)
        p2 = self.maxPoolingLayer(c2)
        #print(p2.shape)
        f = self.flatteningLayer(p2)
        #print(f.shape)
        fc = self.fullyConnectedLayer(f, 'sigmoid', parameters=(self.params["w3"], self.params["b3"]))
        #print(fc.shape)
        fc2 = self.fullyConnectedLayer(fc, 'softmax', parameters=(self.params["w4"], self.params["b4"]))
        #print(fc2.shape)
        cel = self.crossEntropyLoss(fc2, labels)
        print('     cross entropy loss:', cel)

        return (p, p2, fc, fc2, cel)

    def backwardPropagation(self, data, labels, params, forward_prop_func):
        num_examples, _, _, _ = data.shape
        a1, a2, a3, a4, _ = forward_prop_func(data, labels)

        temp = labels - a4
        dj_dw4 = np.dot(a3.T, temp) / -num_examples
        dj_db4 = np.sum(temp, axis=0) / -num_examples

        temp2 = np.dot(temp, self.params["w4"].T) * (a3 * (1 - a3))
        dj_dw3 = (np.dot(a2.T, temp2) / -num_examples).reshape(23760, 300)
        dj_db3 = np.sum(temp2, axis=0) / -num_examples

        #temp3 = np.dot(temp2, self.params["w3"].T).reshape(num_examples, 54, 44, 10) * (a2 * (1 - a2))
        #dj_dw2 = np.dot(a1.T, temp3) / -num_examples
        #dj_db2 = np.sum(temp3, axis=0) / -num_examples

        #print(dj_dw2.shape)
        #print(dj_db2.shape)

        #temp4 = np.dot(temp3, self.params["w2"].T) * (a1 * (1 - a1))
        #dj_dw1 = np.dot(data.T, temp4) / -num_examples
        #dj_db1 = np.sum(temp4, axis=0) / -num_examples

        #print(dj_dw1.shape)
        #print(dj_db1.shape)

        gradients = {
            #"w1" : dj_dw1,
            #"b1" : dj_db1,
            #"w2" : dj_dw2,
            #"b2" : dj_db2,
            "w3" : dj_dw3,
            "b3" : dj_db3,
            "w4" : dj_dw4,
            "b4" : dj_db4
        }

        return gradients
    
    def miniBatchGradientDescent(self, batch_size, backward_prop_func, train_data, train_labels, params, forward_prop_func, learning_rate):
        start_of_batch = 0
        end_of_batch = batch_size

        for i in range(10):
            print(' batch:', i + 1)

            gradients = backward_prop_func(train_data[start_of_batch:end_of_batch], train_labels[start_of_batch:end_of_batch], params, forward_prop_func)

            #dw1 = gradients["w1"]
            #db1 = gradients["b1"]
            #dw2 = gradients["w2"]
            #db2 = gradients["b2"]
            dw3 = gradients["w3"]
            db3 = gradients["b3"]
            dw4 = gradients["w4"]
            db4 = gradients["b4"]

            #params["w1"] = params["w1"] - (learning_rate * dw1)
            #params["b1"] = params["b1"] - (learning_rate * db1)
            #params["w2"] = params["w2"] - (learning_rate * dw2)
            #params["b2"] = params["b2"] - (learning_rate * db2)
            self.params["w3"] = self.params["w3"] - (learning_rate * dw3)
            self.params["b3"] = self.params["b3"] - (learning_rate * db3)
            self.params["w4"] = self.params["w4"] - (learning_rate * dw4)
            self.params["b4"] = self.params["b4"] - (learning_rate * db4)

            start_of_batch += batch_size
            end_of_batch += batch_size

    def convolutionalLayer(self, data, padding=1, stride=1, filter_size=3, num_filters=1, parameters=None):
        num_examples, height, width, num_channels = data.shape

        convolve_filters = parameters[0]
        bias = parameters[1]

        output_height = int(np.floor((height - filter_size + (2 * padding)) / stride) + 1)
        output_width = int(np.floor((width - filter_size + (2 * padding)) / stride) + 1)

        convolved_output = np.zeros((num_examples, output_height, output_width, num_filters))

        for example in range(num_examples):
            exmpl = data[example]

            for filter in range(num_filters):
                conv_temp = np.zeros((output_height, output_width))

                for channel in range(num_channels):
                    image_channel = exmpl[:,:,channel]
                    padded_image_channel = np.pad(image_channel, [(padding, padding), (padding, padding)], mode='constant')
                    conv_temp += self.convolveMatrix2D(padded_image_channel, convolve_filters[:,:,channel,filter])

                conv_temp = conv_temp + bias[filter]
                conv_temp = self.sigmoid(conv_temp)
                convolved_output[example][:,:,filter] = conv_temp

        return convolved_output

    def convolveMatrix2D(self, i, f):
        image = i
        filter = f

        image_h, image_w = image.shape
        filter_size = filter.shape[0]

        output = np.zeros((image_h-2, image_w-2))

        for row in range(image_h-2):
            for pixel in range(image_w-2):
                window = image[row:row+(filter_size), pixel:pixel+(filter_size)]
                output[row][pixel] = np.sum(window * filter)

        return output

    def sigmoid(self, x):
        x = np.clip(x, 0, 1)
        return 1 / (1 + np.exp(-x))

    def maxPoolingLayer(self, data, filter_size=2, stride=2, padding=0):
        num_examples, height, width, num_channels = data.shape

        output_width = int(((width - filter_size + (2 * padding)) / stride) + 1)
        output_height = int(((height - filter_size + (2 * padding)) / stride) + 1)

        pooled_output = np.zeros((num_examples, output_height, output_width, num_channels))

        for example in range(num_examples):
            exmpl = data[example]

            for layer in range(num_channels):
                exmpl_layer = exmpl[:,:,layer]
                max_pool = []

                for row in range(0, height - 1, stride):
                    for col in range(0, width - 1, stride):
                        window = exmpl_layer[row:row + filter_size, col:col + filter_size]
                        window_max = np.amax(window)
                        max_pool.append(window_max)

                max_pool = np.array(max_pool)
                max_pool = np.reshape(max_pool, (output_height, output_width))
                pooled_output[example][:,:,layer] = max_pool

        return pooled_output

    def flatteningLayer(self, data):
        num_examples, depth, height, width = data.shape
        output = []

        for example in range(num_examples):
            output.append(np.reshape(data[example], (depth * height * width)))

        return np.array(output)

    def fullyConnectedLayer(self, data, activation, parameters=None):
        num_examples, _ = data.shape

        weights = parameters[0]
        bias = parameters[1]

        z = np.dot(weights.T, data.T) + np.array([bias for exmpl in range(num_examples)]).T

        if activation=='sigmoid':
            a = self.sigmoid(z)
        elif activation=='softmax':
            a = self.softmax(z)

        return a.T

    def softmax(self, x):
        x = np.clip(x, 0, 218)
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def crossEntropyLoss(self, data, labels):
        num_examples, _ = data.shape
        avg_loss = 0

        for example in range(num_examples):
            avg_loss += -np.dot(labels[example], np.log(data[example]))

        avg_loss = avg_loss / num_examples

        return avg_loss

    def leastSquareLoss(self, data, labels):
        num_examples, _ = data.shape
        avg_loss = 0

        for example in range(num_examples):
            avg_loss += np.power(labels[example] - data[example], 2)

        avg_loss = avg_loss / num_examples

        return avg_loss

def processData():
    train_data = []
    test_data = []
    count = 1

    for image in os.listdir('dataset/img_align_celeba'):
        
        if count <= 100:
            train_data.append(cv.imread('dataset/img_align_celeba/' + image))
        elif count > 100 and count <= 150:
            test_data.append(cv.imread('dataset/img_align_celeba/' + image))
        else:
            break

        count += 1

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    return np.array(train_data), np.array(test_data)

def processLabels():
    train_labels = []
    test_labels = []

    with open('dataset/list_landmarks_align_celeba.csv') as landmarks_dataset_csv:
        image_number = 0

        for row in csv.reader(landmarks_dataset_csv):
            if image_number > 1 and image_number <= 101:
                conv = row[0].replace('   ', ' ')
                conv = conv.replace('  ', ' ')
                conv = conv.split(' ')
                conv = conv[1:]
                conv2 = [int(x) for x in conv]
                conv2 = [1] + conv2
                train_labels.append(conv2)

            elif image_number > 101 and image_number <= 151:
                conv = row[0].replace('   ', ' ')
                conv = conv.replace('  ', ' ')
                conv = conv.split(' ')
                conv = conv[1:]
                conv2 = [int(x) for x in conv]
                conv2 = [1] + conv2
                test_labels.append(conv2)

            image_number += 1

    return np.array(train_labels), np.array(test_labels)

def main():
    train_data, test_data = processData()
    train_labels, test_labels = processLabels()

    cnn = CNN()
    cnn.train(train_data, train_labels)

if __name__ == '__main__':
    main()