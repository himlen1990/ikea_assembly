from hsrb_interface import Robot
import rospy
from hsrb_interface import geometry
from geometry_msgs.msg import WrenchStamped, Wrench
from std_msgs.msg import String, Float32
import numpy



init_flag = False
init_move_flag = False
object_weight = 0
init_counter = 0
motion_finish_flag = False
motion_start_flag = False
force_fb_buffer = []

force_pub = rospy.Publisher('ikea_force_feedback', Float32, queue_size=1)
object_weight_pub = rospy.Publisher('ikea_object_weight', Float32, queue_size=1)

robot = Robot()
whole_body = robot.try_get('whole_body')
gripper = robot.try_get('gripper')
whole_body.end_effector_frame = u'hand_palm_link'
whole_body.linear_weight = 100.0
whole_body.impedance_config = 'compliance_hard'


def move_robot_cb(msg):
    global motion_finish_flag,motion_start_flag
    print msg.data
    if msg.data == "start":
        print "motion_start"
        rospy.sleep(0.5)
        motion_start_flag = True
    if msg.data == "finish":
        print "motion_finish"
        motion_start_flag = False
        motion_finish_flag = True

def force_cb(msg):    
    global init_flag, init_move_flag,  force_fb_buffer, init_counter
    global object_weight, motion_finish_flag, motion_start_flag

    if not init_flag:
        if not init_move_flag:
            whole_body.move_end_effector_pose(geometry.pose(x=0.05), 'hole_location')
            rospy.sleep(1)
            init_move_flag = True
            return            
        if init_counter < 10:
            object_weight = object_weight + msg.wrench.force.x
            init_counter = init_counter + 1
            print msg.wrench.force.x
        elif init_counter >= 10:
            object_weight = object_weight/init_counter
            print "object weight" 
            print object_weight
            print "counter"
            print init_counter
            init_flag = True
            object_weight_pub.publish(object_weight)
            return 

    if init_flag:        
        if motion_start_flag:
            force_fb_buffer.append(msg.wrench.force.x)
        elif motion_finish_flag:
            print "????????"
            print len(force_fb_buffer)
            print min(force_fb_buffer)
            force_pub.publish(min(force_fb_buffer))
            del force_fb_buffer[:]
            print len(force_fb_buffer)
            motion_finish_flag = False

sub = rospy.Subscriber("/hsrb/wrist_wrench/raw", WrenchStamped , force_cb, queue_size=1)
sub = rospy.Subscriber("ikea_motion_signal", String , move_robot_cb , queue_size=1)

rospy.spin()






        
