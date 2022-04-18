'''
y = [0,12,6,8,3,2,10] 
# Convert To String
y = str(y)
# Encode String
y = y.encode()
# Send Encoded String version of the List
s.send(y)



data = connection.recv(4096)
# Decode received data into UTF-8
data = data.decode('utf-8')
# Convert decoded data into list
data = eval(data)
'''
# Import socket module
import socket
import json
import pickle
from threading import local
import zmq



import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint


import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import sys



count = 0
clients = []
n_pixels = 49152
n_classes = 4

lr = 0.01 
comms_round = 100
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               )          


#IMAGE TRAINING

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model





#data will be a list of len 299 where each list has 49152 elements
#labels will be a list of len 299 where each list has 4 elements
def load(paths):
    '''expects images for each class in seperate dir, 
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (224,224))
        #image = np.array(img).flatten()
        if i == 0:
            print(len(image))
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list

        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images

    # return a tuple of the data and labels
    return data, labels

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


#scaling functions
def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    #print("Gloabal Count is ",global_count)
    # get the total number of data points held by a client

    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    #print("Local count is",local_count)
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recvall1(sock):
    data = []
    while True:
        packet = sock.recv(409600)
        if not packet: break
        data.append(packet)
    data_arr = pickle.loads(b"".join(data))
    print (data_arr)
    return data_arr

#wrapper function to receive data
def recvall(sock):
    BUFF_SIZE = 204800 
    data = b''
    weights_list = []
    while True:
        #print("receiving inside fn")
        part = sock.recv(BUFF_SIZE)
        data += part
        #print("Data is ",data)
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    print("end fn")
    return data

# Create a socket object
s = socket.socket()

# Define the port on which you want to connect
port = 50001
s.connect(('127.0.0.1', port))
print("connected at: ", port)

data_dict = {'client_name': 'client1',
             'data_count': 100}

encoded_data = pickle.dumps(data_dict)
s.send(encoded_data)


serverAck = s.recv(1024)



while True:
    
    # request weights message
    print("Client is sending READY")
    s.sendall("READY".encode())


    # receive weights
    #tmp = s.recv(1024)

    #received_weights = json.loads(tmp.decode('utf-8'))
    print("Client is receiving global weights from server ")
    received_weights = pickle.loads(recvall(s))
    #received_weights = recvall(s)
    # Decode received data into UTF-8
    #received_weights = received_weights.decode('utf-8')
    # Convert decoded data into list
    #received_weights = eval(received_weights)
    print("Client has received weights from server")
    print(received_weights[0])

    #break

    # train your images
    #declear path to your mnist data folder
 
     #img_path = 'client1_train'
    #get the path list using the path object
    # image_paths = list(paths.list_images(img_path))

    #apply our function
    # image_list, label_list = load(image_paths)

    #binarize the labels
    # lb = LabelBinarizer()
    # label_list = lb.fit_transform(label_list)

    #split data into training and test set
    # X_train = image_list
    # y_train = label_list
    # print("X train[0] length is ",len(X_train[0]))
    # print("Y train[0] length is ",len(y_train[0]))
    # X_train, X_test, y_train, y_test = train_test_split(image_list, 
    #                                                     label_list, 
    #                                                     test_size=0.1, 
    #                                                     random_state=42)


    # smlp_local = SimpleMLP()
    # local_model = smlp_local.build(n_pixels, n_classes)
    # local_model.compile(loss=loss, 
    #                   optimizer=optimizer, 
    #                   metrics=metrics)
    #local_model.set_weights(received_weights)
    # local_model.fit(X_train, y_train,batch_size = 64,epochs=1)
   

    IMAGE_SIZE = [224, 224]

    train_path = 'client2_train'

    vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    for layer in vgg.layers:
        layer.trainable = False


    x = Flatten()(vgg.output)

    prediction = Dense(4, activation='softmax')(x)

    model = Model(inputs=vgg.input, outputs=prediction)
    model.set_weights(received_weights)


    model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )


    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                    target_size = (224, 224),
                                                    batch_size = 4,
                                                    class_mode = 'categorical')

    checkpointer =ModelCheckpoint(filepath = 'client1.hdf5',verbose=1,save_best_only=True)

    print("Client is training ")

    # r = model.fit(
    #     training_set,
    #     epochs=1,
    #     callbacks= [checkpointer],
    #     steps_per_epoch=len(training_set),
       
    # )
    print("Training ended on client")
    tf.keras.models.save_model(model, 'client1.hdf5')
    print("Client Weights saved")
    # x = 0
    # for i in range(100000):
    #     x+=i

    # test_set = test_datagen.flow_from_directory('test_set',
    #                                             target_size = (224, 224),
    #                                             batch_size = 32,
    #                                             class_mode = 'categorical')

    #scale the model weights and add to list
    scaling_factor = 0.5
    #for now, not scaling weights, converting to list format for pickling
    scaled_weights = scale_model_weights(model.get_weights(),scaling_factor)
    #scaled_local_weight_list.append(scaled_weights)
    
    #encoded_data = json.dumps(global_weights).encode('utf-8')
    #clientsocket.sendall(bytes('ò', global_model.get_weights()))

    trained_weights = model.get_weights()
    print("trained weights dump start")
    data=pickle.dumps(scaled_weights)
    #local_weights = [l.tolist() for l in local_weights]
    print("trained weights dump end")
    s.sendall(data)
    print("client has sent the trained weights")
    serverAck = s.recv(1024)
    print(serverAck)



#s.close()
