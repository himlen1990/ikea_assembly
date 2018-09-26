#!/usr/bin/env python

import roslib
roslib.load_manifest('ikea_ar')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    #self.image_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color",Image,self.callback)
    self.image_sub = rospy.Subscriber("/camera/rgb/image_rect_color",Image,self.callback)
    self.count = 0
    self.save_dir = './dataset'
    #self.save_dir = './object_dataset'



  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    #(rows,cols,channels) = cv_image.shape
    #if cols > 60 and rows > 60 :
    #  cv2.circle(cv_image, (50,50), 10, 255)

    crop_img = cv_image[:448,320-224:320+224]
    cv2.imshow("cropped image", crop_img)
    #cv2.imshow("Image window2", cv_image)
    k = cv2.waitKey(3) & 0xFF
    if k == ord('s'):
      self.image_name = self.save_dir + '/' + 'frame%04d' %(self.count) + '.jpg'
      cv2.imwrite(self.image_name, crop_img)      
      print "saved image: ", self.image_name
      self.count = self.count + 1



def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
