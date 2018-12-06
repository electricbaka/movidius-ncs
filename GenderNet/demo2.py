#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# https://github.com/movidius/ncappzoo

# Copyright(c) 2018 JellyWare Inc.
# License: MIT
# https://github.com/electricbaka/movidius-ncs

import sys
import numpy
import cv2
from mvnc import mvncapi as mvnc

#============================================================
# Set up
#============================================================
#------------------------------
# NCS
#------------------------------
# configure the NCS
mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

# Get a list of ALL the sticks that are plugged in
devices = mvnc.enumerate_devices()
if len(devices) == 0:
	print('No devices found')
	quit()

# Pick the first stick to run the network
device = mvnc.Device(devices[0])

# Open the NCS
device.open()

# open the network blob files
blob='graph'

# Load blob
with open(blob, mode='rb') as f:
	blob = f.read()
graph = mvnc.Graph('graph')
fifoIn, fifoOut = graph.allocate_with_fifos(device, blob)

#------------------------------
# GenderNet
#------------------------------
# categories for age and gender
age_list=['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
gender_list=['Male','Female']

# read in and pre-process the image:
ilsvrc_mean = numpy.load('../../data/age_gender/age_gender_mean.npy').mean(1).mean(1) #loading the mean file
dim=(227,227)

#------------------------------
# Capture and FullScreen
#------------------------------
cap = cv2.VideoCapture(0)

img1 = cv2.imread("car1366.jpg")
img2 = cv2.imread("cake1366.jpg")
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
img_out = numpy.zeros((768, 1366, 3), numpy.uint8)

#============================================================
# Main Loop
#============================================================
while True:
	ret, frame = cap.read()

	# Reload on error
	if ret == False:
		continue

	# Wait key
	key = cv2.waitKey(1)
	if key != -1:
		break

	#------------------------------
	# GenderNet prediction
	#------------------------------
	img = frame.copy()
	img = cv2.resize(img,dim)
	img = img.astype(numpy.float32)
	img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
	img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
	img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

	graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img, 'user object')
	output, userobj = fifoOut.read_elem()

	print('\n------- predictions --------')
	order = output.argsort()
	last = len(order)-1
	predicted=int(order[last])
	print('the predicted gender is ' + gender_list[predicted] + ' with confidence of %3.1f%%' % (100.0*output[predicted]))

	#------------------------------
	# Display
	#------------------------------
	# Select FullScreen Image
	if predicted == 0:
		img_out = img1
	elif predicted == 1:
		img_out = img2

	# Picture in Picture
	img_camera =  cv2.resize(frame,(400,300))
	img_display = img_out.copy()
	img_display[20:320, 20:420] = img_camera

	# Add Text
	text = '%3.1f%%' % (100.0*output[predicted]) + ' ' + gender_list[predicted]
	cv2.putText(img_display, text, (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 4)

	# Display
	cv2.imshow("window", img_display)

#============================================================
# Clean up
#============================================================
# graph and the device
fifoIn.destroy()
fifoOut.destroy()
graph.destroy()
device.close()

# Capture
cap.release()
cv2.destroyAllWindows()
