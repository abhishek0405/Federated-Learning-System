# Import socket module
import socket
import json

# Create a socket object
s = socket.socket()

# Define the port on which you want to connect
port = 50001
s.connect(('127.0.0.1', port))

data_dict = {'client_name': 'client1',
             'data_count': 100}

encoded_data = json.dumps(data_dict).encode('utf-8')
s.sendall(encoded_data)
while True:

    # request weights message

    # receive weights

    # train your images

    # send weights

    # ---------------------------------------------------

    # if(msg == "X"):
    #     break

    # s.send(msg.encode())

    # reply = s.recv(1024).decode()

    # print("Reply: ", reply)

s.close()
