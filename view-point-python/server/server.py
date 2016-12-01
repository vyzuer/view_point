import socket               # Import socket module
import struct
import Image
import io
from threading import Thread
import sys

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-python')

import prediction.recommendation as recsys


def receive_data(conn):
    # receive lat and long
    buf = ''
    while len(buf) < 8:
        buf += conn.recv(8 - len(buf))
    lat = struct.unpack('!d', buf)[0]
    print 'lat received'
    buf = ''
    while len(buf) < 8:
        buf += conn.recv(8 - len(buf))
    lon = struct.unpack('!d', buf)[0]
    print 'lon received'
    print lat, lon
    
    # receive the image from the client
    # first read the size of the image frame
    buf = ''
    while len(buf) < 4:
        buf += conn.recv(4 - len(buf))
    size = struct.unpack('!i', buf)[0]
    print size
    
    i = 0
    byteArray = bytearray(size)
    while size > 0:
        data = conn.recv(1024)
        width = len(data)
        byteArray[i:i+width] = data
        size -= len(data)
    
        i = i+width
    
    size = len(byteArray)
    print size
    
    image = Image.open(io.BytesIO(byteArray))
    
    image.save("/home/vyzuer/Desktop/image.jpg", "JPEG")
    print 'image saved.'

    return lat, lon, image


def send_data(conn, lat, lon, image):
    # send the resulted lat and long
    data = struct.pack('>d', lat)
    conn.send(data)
    print 'lat sent'
    data = struct.pack('>d', lon)
    conn.send(data)
    print 'lon sent'

    byteArray = None
    with open("/home/vyzuer/Desktop/image.jpg", "rb") as imageFile:
        f = imageFile.read()
        byteArray = bytearray(f)

    size = len(byteArray)
    # first send the size of image
    num_bytes = struct.pack('!i', size)
    conn.send(num_bytes)
    print 'size sent'
    conn.send(byteArray)
    print 'image sent'


def talk_to_client(conn):
    # communicate with the client and exit
    lat, lon, image = receive_data(conn)

    # generate the recommenations

    # send the result back to the server
    send_data(conn, lat, lon, image)


def start_server():

    # create a socket for connection
    soc = socket.socket()         # Create a socket object
    # host = "localhost" # Get local machine name
    host = "172.26.186.159" # Get local machine name
    port = 2004                # Reserve a port for your service.
    soc.bind((host, port))       # Bind to the port
    soc.listen(5)                 # Now wait for client connection.

    while True:
        # listen for incoming connections
        conn, addr = soc.accept()     # Establish connection with client.
        print ("Got connection from",addr)

        # now start a new thread for this client
        client_thread = Thread(target=talk_to_client, args=(conn, ))
        client_thread.start()


if __name__ == "__main__":
    
    start_server()

