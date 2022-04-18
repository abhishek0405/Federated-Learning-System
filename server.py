# # first of all import the socket library
# import socket
# # next create a socket object
# s = socket.socket()
# print("Socket successfully created")
# port = 12345
# s.bind(('', port))
# print("socket binded to %s" % (port))
# s.listen(5)
# print("socket is listening")

# c, addr = s.accept()
# print('Got connection from', addr)
# while True:
#     c.setblocking(0)
#     c, addr = s.accept()

#     c.setblocking(1)
#     # Establish connection with client.

#     msg = c.recv(1024).decode()

#     print("message: ", msg)

#     reply = input("Enter your reply ")

#     if(reply == "X"):
#         break

#     c.send(reply.encode())




# Use pickle or json to send list(or any other object for that matter) over sockets depending on the receiving side. You don't need json in this case as your receiving host is using python.

# import pickle
# y=[0,12,6,8,3,2,10] 
# data=pickle.dumps(y)
# s.send(data)
# Use pickle.loads(recvd_data) on the receiving side.

#!/usr/bin/python           # This is server.py file

import socket               # Import socket module
import threading
import _thread
import json
import pickle
import sys
import zmq


#tensorflow imports
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


# create a global weights array
count = 0
clients = []
n_pixels = 49152
n_classes = 4
scaled_local_weight_list = list()


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


IMAGE_SIZE = [224, 224]

test_path = 'Data/test'

vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False


x = Flatten()(vgg.output)

prediction = Dense(4, activation='softmax')(x)

# global_model = Model(inputs=vgg.input, outputs=prediction)
global_model = tf.keras.models.load_model('client1.hdf5')




#GLOBAL MODEL INITIALIZE
#smlp_global = SimpleMLP()
#global_model = smlp_global.build(n_pixels, n_classes)

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


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = []
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def on_new_client(clientsocket, addr, count):
    print("received a connection")
    # check if better way to receive for larger data
    received = pickle.loads(recvall(clientsocket))
    print(received)
    if not any(d['client_name'] == received['client_name'] for d in clients):

            cl = {'client_name': received['client_name'], 'socket': clientsocket, 'addr': addr} #cl['addr'][1] is port
            clients.append(cl)
            print(cl['addr'][1])
    
    clientsocket.send("Ack".encode())
        
    while True:

        
        
        #recv = json.loads(tmp)
        #print(recv[0])
        #received = recv[0]
        # add the client details to a dict
        
        
        #ack
      
        print("Server is ready to receive READY Request")
        # receive weights request from client
        weight_req = clientsocket.recv(1024).decode()
        print(weight_req)
        if weight_req == 'READY':

            # send global weights
            
            global_weights = global_model.get_weights()
            scaled_global_weights = scale_model_weights(global_weights,1)
            #global_weights = [l.tolist() for l in global_weights]
            #global_weights = str(global_weights)
            #global_weights = global_weights.encode()
            #print(global_weights)
            print(sys.getsizeof(global_weights))
            encoded_data = pickle.dumps(scaled_global_weights)
            clientsocket.sendall(encoded_data)
            #data=pickle.dumps(global_weights)
            #clientsocket.send(data)


            # receive trained weights
            print("Server is receiving trained weights from client")
            #trained_weights = pickle.loads(recvall(clientsocket))
            trained_weights = pickle.loads(recvall(clientsocket))
            #trained_weights = recvall1(clientsocket)
            print("Server has received trained weights from client")
            print(trained_weights[0])
            count += 1

            # compute avg -> (global + received)/2 ie global updated
            if count == 1:
                global_model.set_weights(trained_weights)
                print("Server is updating weights")
            else:

                scaling_factor = 0.5
                scaled_global_weights = scale_model_weights(global_model.get_weights(), scaling_factor)
                scaled_curr_weights = scale_model_weights(trained_weights, scaling_factor)
                scaled_weight_list = []
                scaled_weight_list.append(scaled_global_weights)
                scaled_weight_list.append(scaled_curr_weights)
                average_weights = sum_scaled_weights(scaled_weight_list)
        
                #update global model 
                global_model.set_weights(average_weights)
            print("round completed at server")
            clientsocket.send("Ack".encode())



        #compute test score
        #if test score reaches a threshold, then break
        #or if total loops>10 then break

        # --------------------------------------------------

        
    #clientsocket.close()

print('Server started!')

mode = input("Enter TRAIN if you want to train the system else enter TEST  ")

if(mode=="TRAIN"):

    s = socket.socket()         # Create a socket object
    host = socket.gethostname()  # Get local machine name
    port = 50001                # Reserve a port for your service.


    print('Waiting for clients...')

    s.bind(('', port))        # Bind to the port
    s.listen(5)                 # Now wait for client connection.


    while True:
        c, addr = s.accept()     # Establish connection with client.
        _thread.start_new_thread(on_new_client, (c, addr, count))
        #count += 1
        # Note it's (addr,) not (addr) because second parameter is a tuple
        #Edit: (c,addr)
        # that's how you pass arguments to functions when creating new threads using thread module.


    s.close()

elif mode=="TEST":
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

    score= global_model.evaluate(test_set)

    print("Testing score is ",score)

    #confusion matrix
    # batch_size = 64
    # target_names = category_list
    # Y_pred = model_dilated.predict(test_batch, 15006 // batch_size+1)
    # y_pred = np.argmax(Y_pred, axis=1)
    # print('Confusion Matrix')
    # cm = metrics.confusion_matrix(test_batch.classes, y_pred)

else:
    print("Enter a valid query")

