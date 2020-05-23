#!/usr/bin/python
#@uthor : Sumanth Nethi
import pyrealsense2 as rs
import numpy as np
import cv2 
import matplotlib.pyplot as plt

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("/home/sumanth/librealsense/build/examples/scripts/test.bag")
profile = pipe.start(cfg)

for x in range(50):      #to get a well-exposed image
  pipe.wait_for_frames()
       
frame = pipe.wait_for_frames()
color_frame = frame.get_color_frame()
depth_frame = frame.get_depth_frame()
pipe.stop()

lowerBound=np.array([39,100,31]) #hsv limits to detect the book
upperBound=np.array([51,198,76])

color = np.asanyarray(color_frame.get_data())
align = rs.align(rs.stream.color)
frame = align.process(frame)

aligned_depth_frame = frame.get_depth_frame()
colorized_depth = np.asanyarray(rs.colorizer().colorize(aligned_depth_frame).get_data())

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv,lowerBound,upperBound)
maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5)))
maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,np.ones((20,20)))
maskFinal = maskClose
conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

x,y,w,h = cv2.boundingRect(conts[0])
cv2.rectangle(color,(x,y),(x+w,y+h),(0,0,255),3)
cv2.rectangle(colorized_depth,(x,y),(x+w,y+h),(0,0,255),3)
depth = np.asanyarray(aligned_depth_frame.get_data())
depth = depth[x:x+w,y:y+h].astype(float)
dpt = np.mean(depth)
print("Depth of the book : "),
print(dpt)
cv2.imshow('Depth',colorized_depth)
cv2.imshow('Image', color)
cv2.waitKey(0)












