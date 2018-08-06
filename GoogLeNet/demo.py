#! /usr/bin/env python3

from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys

dim=(224,224)
EXAMPLES_BASE_DIR='../../'

# ***************************************************************
# get labels
# ***************************************************************
labels_file=EXAMPLES_BASE_DIR+'data/ilsvrc12/synset_words.txt'
labels=numpy.loadtxt(labels_file,str,delimiter='\t')

# ***************************************************************
# configure the NCS
# ***************************************************************
mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

# ***************************************************************
# Get a list of ALL the sticks that are plugged in
# ***************************************************************
devices = mvnc.enumerate_devices()
if len(devices) == 0:
	print('No devices found')
	quit()

# ***************************************************************
# Pick the first stick to run the network
# ***************************************************************
device = mvnc.Device(devices[0])

# ***************************************************************
# Open the NCS
# ***************************************************************
device.open()

network_blob='graph'

#Load blob
with open(network_blob, mode='rb') as f:
	blob = f.read()

graph = mvnc.Graph('graph')
fifoIn, fifoOut = graph.allocate_with_fifos(device, blob)

# ***************************************************************
# video capture
# ***************************************************************
detect_counter = 0
detect_old = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('image', frame)
    key = cv2.waitKey(1)
    if key != -1:
        break

    # ***************************************************************
    # Load the image
    # ***************************************************************
    ilsvrc_mean = numpy.load(EXAMPLES_BASE_DIR+'data/ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1) #loading the mean file
    #img = cv2.imread(EXAMPLES_BASE_DIR+'data/images/nps_electric_guitar.png')
    img = frame
    img=cv2.resize(img,dim)
    img = img.astype(numpy.float32)
    img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
    img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
    img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img, 'user object')

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = fifoOut.read_elem()

    # ***************************************************************
    # Print the results of the inference form the NCS
    # ***************************************************************
    order = output.argsort()[::-1][:6]
    print('\n------- predictions --------')
    for i in range(0,5):
    	print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + labels[order[i]] + '  label index is: ' + str(order[i]) )

    # ***************************************************************
    # detect and sound
    # ***************************************************************
    #detect
    detect_new = order[0]
    if detect_new == detect_old:
        detect_counter = detect_counter + 1
    else:
        detect_counter = 0

    detect_old = detect_new

    #sound
    if detect_counter == 2:
        if detect_new == 546:
            os.system('aplay ./sound1.wav &')
        elif detect_new == 402:
            os.system('aplay ./sound2.wav &')
        elif detect_new == 508:
            os.system('aplay ./sound3.wav &')

# ***************************************************************
# Clean up video capture and windows
# ***************************************************************
cap.release()
cv2.destroyAllWindows()

# ***************************************************************
# Clean up the graph and the device
# ***************************************************************
fifoIn.destroy()
fifoOut.destroy()
graph.destroy()
device.close()
