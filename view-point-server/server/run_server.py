import socket               # Import socket module
import struct
import Image
import io
from threading import Thread
import sys
import atexit
import traceback, os.path
import linecache
import numpy as np

# add the package
sys.path.append('/home/vyzuer/Copy/Research/Project/code/view-point/view-point-server')
sys.path.append('/home/vyzuer/Copy/Research/Project/code/group-photography/gpa_package_server/lib')

import prediction.recommendation as recsys
import server_code.gp_recommendation as gp_rec

def close_server():
    pass

def receive_mode(conn):
    buf = ''
    while len(buf) < 4:
        buf += conn.recv(4 - len(buf))
    mode = struct.unpack('!i', buf)[0]
    print mode

    return mode
    

def readImage(conn):
    # receive the image from the client
    # first read the size of the image frame
    size = readInt(conn)
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

    # resize image
    image = resize(image)
    
    image.save("/home/vyzuer/Desktop/image.jpg", "JPEG")
    print 'image saved.'

    return image

def resize(image):
    if image.width > 640 or image.height > 640:
        if image.height > image.width:
            factor = 640.0/image.height
        else:
            factor = 640.0/image.width

        h = int(image.height * factor)
        w = int(image.width * factor)
        print h , w
        image = image.resize((w,h), Image.ANTIALIAS)
        print factor

    print image.height, image.width

    return image

def readFloat(conn):
    buf = ''
    while len(buf) < 8:
        buf += conn.recv(8 - len(buf))
    value = struct.unpack('!d', buf)[0]

    return value

def readInt(conn):
    buf = ''
    while len(buf) < 4:
        buf += conn.recv(4 - len(buf))
    data = struct.unpack('!i', buf)[0]

    return data

def receive_data_gp(conn):

    # read num people
    num_people = readInt(conn)
    print 'num_people received'

    color_data = np.zeros(shape=(num_people, 3))

    for i in range(num_people):
        color_data[i][0] = readFloat(conn)
        color_data[i][1] = readFloat(conn)
        color_data[i][2] = readFloat(conn)

        print color_data[i]

    print 'colors received'
    
    image = readImage(conn)

    return num_people, color_data, image

def receive_data_pa(conn):
    # receive lat and long
    lat = readFloat(conn)
    print 'lat received'

    lon = readFloat(conn)
    print 'lon received'
    print lat, lon
    
    image = readImage(conn, size)

    return lat, lon, image

def receive_data_vp(conn):
    # receive lat and long
    lat = readFloat(conn)
    print 'lat received'

    lon = readFloat(conn)
    print 'lon received'
    print lat, lon
    
    image = readImage(conn, size)

    return lat, lon, image

def sendInt(conn, value):
    data = struct.pack('!i', value)
    conn.send(data)

def sendFloat(conn, value):
    data = struct.pack('>d', value)
    conn.send(data)


def send_data_gp(conn, num_people, rec_pos, color_data):

    sendInt(conn, num_people)
    print 'num_people sent'

    for i in range(num_people):
        sendInt(conn, rec_pos[i][0])
        sendInt(conn, rec_pos[i][1])
        sendInt(conn, rec_pos[i][2])
        sendInt(conn, rec_pos[i][3])

        print rec_pos[i]

    # send color information
    for i in range(num_people):
        sendFloat(conn, color_data[i][0])
        sendFloat(conn, color_data[i][1])
        sendFloat(conn, color_data[i][2])

        print color_data[i]


def send_data_vp(conn, lat, lon, image):
    # send the resulted lat and long
    sendFloat(conn, lat)
    print 'lat sent'
    sendFloat(conn, lon)
    print 'lon sent'

    byteArray = None
    with open("/home/vyzuer/Desktop/image.jpg", "rb") as imageFile:
        f = imageFile.read()
        byteArray = bytearray(f)

    size = len(byteArray)
    # first send the size of image
    sendInt(conn, size)
    print 'size sent'
    conn.send(byteArray)
    print 'image sent'


def send_data_pa(conn, lat, lon, image):
    # send the resulted lat and long
    sendFloat(conn, lat)
    print 'lat sent'
    sendFloat(conn, lon)
    print 'lon sent'

    byteArray = None
    with open("/home/vyzuer/Desktop/image.jpg", "rb") as imageFile:
        f = imageFile.read()
        byteArray = bytearray(f)

    size = len(byteArray)
    # first send the size of image
    sendInt(conn, size)
    print 'size sent'
    conn.send(byteArray)
    print 'image sent'


def communicate_gp(conn):
    rec_pos = None

    # communicate with the client and exit
    num_people, color_data, image = receive_data_gp(conn)

    # generate the recommenations
    try:
        num_people, rec_pos, color_data = gp_rec.gen_recommendation(np.asarray(image), num_people, color_data)
    except Exception as e:        
        traceback.print_exc()
        print 'Failed to generate recommendation: %s' % e

    # send the result back to the server
    send_data_gp(conn, num_people, rec_pos, color_data)

def talk_to_client(conn):
    # first receive the mode
    mode = receive_mode(conn)

    if mode == 0:
        communicate_pa(conn)
        pass
    elif mode == 1:
        communicate_vp(conn)
    else:
        communicate_gp(conn)


def communicate_pa(conn):
    # communicate with the client and exit
    lat, lon, image = receive_data_pa(conn)

    r_lat = 0.0
    r_lon = 0.0
    img_src = image

    # generate the recommenations
    try:
        r_lat, r_lon, img_src = recsys.recommendation(np.asarray(image), lat, lon)
    except Exception as e:        
        traceback.print_exc()
        print 'Failed to generate recommendation: %s' % e

    # send the result back to the server
    send_data_pa(conn, r_lat, r_lon, img_src)

def communicate_vp(conn):
    # communicate with the client and exit
    lat, lon, image = receive_data_vp(conn)

    r_lat = 0.0
    r_lon = 0.0
    img_src = image

    # generate the recommenations
    try:
        r_lat, r_lon, img_src = recsys.recommendation(np.asarray(image), lat, lon)
    except Exception as e:        
        traceback.print_exc()
        print 'Failed to generate recommendation: %s' % e

    # send the result back to the server
    send_data_vp(conn, r_lat, r_lon, img_src)


def start_server():

    # create a socket for connection
    soc = socket.socket()         # Create a socket object
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # host = "localhost" # Get local machine name
    host = "172.26.186.159" # Get local machine name
    port = 2004                # Reserve a port for your service.
    soc.bind((host, port))       # Bind to the port
    soc.listen(5)                 # Now wait for client connection.

    while True:
        # listen for incoming connections
        print 'waiting for client...'
        conn, addr = soc.accept()     # Establish connection with client.
        print ("Got connection from",addr)

        # now start a new thread for this client
        client_thread = Thread(target=talk_to_client, args=(conn, ))
        client_thread.start()


if __name__ == "__main__":
    
    atexit.register(close_server)

    print 'Starting Server.....\n'

    start_server()

