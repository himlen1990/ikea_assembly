#!/usr/bin/env python

import roslib
roslib.load_manifest('ikea_ar')
import sys
import rospy
from geometry_msgs.msg import PointStamped, PoseArray
import numpy as np
import tf


location_update_flag = False

#msg_counter = 0
x = []
y = []
z = []

cx = []
cy = []
cz = []


def callback(msg):
  global location_update_flag
  global x,y,z,cx,cy,cz
  print "get_new_message"

  #msg_counter = msg_counter + 1
  x = []
  y = []
  z = []
  cx = []
  cy = []
  cz = []

  for i in range(len(msg.poses)):
    x.append(msg.poses[i].position.x)
    y.append(msg.poses[i].position.y) 
    z.append(msg.poses[i].position.z)
  location_update_flag = True

def send_tf():
  global location_update_flag
  global x,y,z,cx,cy,cz
  get_result_flag = False
  rospy.init_node('send_tf', anonymous=True)
  #sub = rospy.Subscriber("hole", PointStamped , callback)
  sub = rospy.Subscriber("holes", PoseArray , callback)
  rate = rospy.Rate(10)
  br = tf.TransformBroadcaster()
  listener = tf.TransformListener()

  while not rospy.is_shutdown():
    try:
      (trans,rot) = listener.lookupTransform('/map','/head_rgbd_sensor_link',rospy.Time(0))  
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      continue

    if location_update_flag and trans and rot:
      trans_mat = tf.transformations.translation_matrix(trans)
      rot_mat   = tf.transformations.quaternion_matrix(rot)
      mat = np.dot(trans_mat, rot_mat)        
      for i in range(len(x)):
        object_to_rgbd = np.identity(4)
        object_to_rgbd[0][3] = x[i]
        object_to_rgbd[1][3] = y[i]
        object_to_rgbd[2][3] = z[i]      
        object_to_base = np.dot(mat,object_to_rgbd)        
        quat = tf.transformations.quaternion_from_euler(3.14,-1.57,0) 
        frame_name = "hole%d" %(i)
        #frame_name = "hole_location" 
        br.sendTransform((object_to_base[0][3],object_to_base[1][3],object_to_base[2][3]+0.15),
                         quat,
                         rospy.Time.now(),
                         frame_name,
                         "map")

        cx.append(object_to_base[0][3])
        cy.append(object_to_base[1][3])
        cz.append(object_to_base[2][3])
        get_result_flag = True
        location_update_flag = False

    if get_result_flag and not location_update_flag:
      for i in range(len(cx)):
        frame_name = "hole%d" %(i)
        quat = tf.transformations.quaternion_from_euler(3.14,-1.57,0)        
        br.sendTransform((cx[i],cy[i],cz[i]+0.15),
                         quat,
                         rospy.Time.now(),
                         frame_name,
                         "map")

    rate.sleep()

if __name__ == '__main__':
    try:
        send_tf()
    except rospy.ROSInterruptException:
        pass
