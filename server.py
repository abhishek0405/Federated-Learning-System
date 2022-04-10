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


#!/usr/bin/python           # This is server.py file

import socket               # Import socket module
import threading
import _thread
import json

# create a global weights array


def on_new_client(clientsocket, addr):
    print("received a connection")
    while True:

        # check if better way to receive for larger data
        tmp = clientsocket.recv(1024)

        d = json.loads(tmp.decode('utf-8'))
        print(d)
        # add the client details to a dict

        # receive weights request from client

        # send global weights

        # receive trained weights

        # compute avg -> (global + received)/2 ie global updated

        # --------------------------------------------------

        # msg = clientsocket.recv(1024).decode()
        # # do some checks and if msg == someWeirdSignal: break:
        # print(addr, msg)
        # msg = input('SERVER >> ')
        # # Maybe some code to compute the last digit of PI, play game or anything else can go here and when you are done.
        # clientsocket.send(msg.encode())
    clientsocket.close()


s = socket.socket()         # Create a socket object
host = socket.gethostname()  # Get local machine name
port = 50001                # Reserve a port for your service.

print('Server started!')
print('Waiting for clients...')

s.bind(('', port))        # Bind to the port
s.listen(5)                 # Now wait for client connection.


while True:
    c, addr = s.accept()     # Establish connection with client.
    _thread.start_new_thread(on_new_client, (c, addr))
    # Note it's (addr,) not (addr) because second parameter is a tuple
    #Edit: (c,addr)
    # that's how you pass arguments to functions when creating new threads using thread module.


s.close()
