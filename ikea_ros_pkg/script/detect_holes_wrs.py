#!/usr/bin/env python

import roslib
roslib.load_manifest('ikea_ar')
import sys
import rospy
import cv2
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow as tf
from model import regression_model
import message_filters
import tf as ros_tf
import math

device = "hsr" # "astra" or "hsr"

class hole_detector:

  def __init__(self):
    self.bridge = CvBridge()    

    #self.point_pub = rospy.Publisher("/hole", PointStamped, queue_size = 5)
    self.poses_pub = rospy.Publisher("/holes", PoseArray, queue_size = 5)

    if device == "astra":
      self.rgb_sub = message_filters.Subscriber('/camera/rgb/image_rect_color', Image)
      self.depth_sub = message_filters.Subscriber('/camera/depth_registered/image_raw', Image)
      self.cam_info_sub = rospy.Subscriber("/camera/rgb/camera_info",CameraInfo,self.cam_info_callback)
    elif device == "hsr":
      self.rgb_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_rect_color', Image)
      self.depth_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw', Image)
      self.cam_info_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/camera_info",CameraInfo,self.cam_info_callback)
      
    self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
    self.ts.registerCallback(self.callback)


    self.save_dir = './dataset'
    model_graph = tf.Graph()
    sess_a = tf.Session(graph=model_graph)

    
    with sess_a.as_default():
      with model_graph.as_default():
        self.model = regression_model(sess_a)
        saver = tf.train.Saver()
        saver.restore(sess_a, "./models/params")

    self.cell_size = 8
    self.boundary = self.cell_size * self.cell_size
    

  def cam_info_callback(self, msg):
    self.fx = msg.K[0]
    self.fy = msg.K[4]
    self.cx = msg.K[2]
    self.cy = msg.K[5]
    self.invfx = 1.0/self.fx
    self.invfy = 1.0/self.fy
    self.cam_info_sub.unregister()

  def unproject(self,u,v,z):
      x = (u - self.cx) * z * self.invfx
      y = (v - self.cy) * z * self.invfy
      return x,y

      
  def callback(self,rgb, depth):
    try:
      rgb_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
    except CvBridgeError as e:
      print(e)

    try:
      depth_image = self.bridge.imgmsg_to_cv2(depth, "passthrough")
    except CvBridgeError as e:
      print(e)

    
    crop_img = rgb_image[:448,320-224:320+224] #bias
    crop_img_depth = depth_image[:448,320-224:320+224]

    k = cv2.waitKey(3) & 0xFF
    if k == ord('s'):
      #dest_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) for skimage training
      #result = self.model.predict(dest_img)
      result = self.model.predict(crop_img)
      mask = np.reshape(result[:,:self.boundary],[self.cell_size,self.cell_size])
      x = np.reshape(result[:,self.boundary:self.boundary*2],[self.cell_size,self.cell_size])
      y = np.reshape(result[:,self.boundary*2:],[self.cell_size,self.cell_size])
      restruct_label = np.stack((mask,x,y),axis=-1)
      
      x2D,y2D = self.show_label(crop_img,restruct_label)
      locations = PoseArray()
      if device == "astra":
        locations.header.frame_id = "/camera_rgb_optical_frame"
      elif device == "hsr":
        locations.header.frame_id = "/head_rgbd_sensor_link"

      locations.header.stamp = rospy.Time.now()
      x3D_buffer = []
      y3D_buffer = []
      z_buffer = []
      distance_to_cam = []
      for i in range(len(x2D)):
        recover_x = x2D[i] + 320 - 224 #recover from cropped image
        z = depth_image[y2D[i],recover_x]
        if device == "hsr":
          z = z/1000.0 #in hsr
        if (z > 0 and hasattr(self,'fx')):   #compute 3D pose
          x3D,y3D = self.unproject(recover_x,y2D[i],z)
          pose = Pose()
          pose.position.x = x3D
          pose.position.y = y3D
          pose.position.z = z
          locations.poses.append(pose)
      self.poses_pub.publish(locations)
    cv2.imshow("Image window", crop_img)



  def show_label(self, image, predict):
    mask = predict[...,0]
    cell_length = image.shape[0]/mask.shape[0]
    x = predict[...,1]
    y = predict[...,2]
    mask_f = mask.flatten()
    top_n = 4
    n_max = mask_f.argsort()[-top_n:]    
    print mask_f[n_max]
    top_n_idx = []
    temp_x = []
    temp_y = []
    dis_to_cam = []
    sorted_x = []
    sorted_y = []
    for i in range(top_n):
      print mask_f[n_max[i]]
      if mask_f[n_max[i]] > 0.0:
        idx = np.unravel_index(n_max[i],mask.shape)
        top_n_idx.append(idx)

    for i in range(0, len(top_n_idx)):
        circle_x = int(x[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][1]*cell_length)
        circle_y = int(y[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][0]*cell_length)
        cv2.circle(image,(circle_x,circle_y),3,(0,255,0))
        temp_x.append(circle_x)
        temp_y.append(circle_y)
        dis_to_cam.append(math.sqrt((448-circle_x)*(448-circle_x) + (448-circle_y)*(448-circle_y)))
    #print temp_x
    #print temp_y
    
    index = [dis_to_cam.index(x) for x in sorted(dis_to_cam,reverse=True)]
    print index
    for i in range(len(temp_x)): #sort from far to close
      sorted_x.append(temp_x[index[i]])
      sorted_y.append(temp_y[index[i]])
    print "sorted"
    print sorted_x
    print sorted_y

    cv2.imshow("test",image)

    return sorted_x, sorted_y




def main(args):
  rospy.init_node('image_converter', anonymous=True)
  hd = hole_detector()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
