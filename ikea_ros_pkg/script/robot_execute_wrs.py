from hsrb_interface import Robot
import rospy
from hsrb_interface import geometry
from geometry_msgs.msg import WrenchStamped, Wrench
from std_msgs.msg import String, Float32


step = 0.005
init_z = -0.018
init_y = 0
next_z = init_z
next_y = init_y
object_weight = -1
z_counter = 0
y_counter = 0
positive_dir = True
negative_dir = False
fail_flag = False

hole_id = 0

robot = Robot()
whole_body = robot.try_get('whole_body')
gripper = robot.try_get('gripper')
whole_body.end_effector_frame = u'hand_palm_link'
whole_body.linear_weight = 100.0
whole_body.impedance_config = 'compliance_hard'
motion_signal_pub = rospy.Publisher('ikea_motion_signal', String, queue_size=1)

def object_weight_cb(msg):
    global object_weight,hole_id
    object_weight = msg.data
    print "object_weight",object_weight
    hole_frame_name = 'hole%d' %(hole_id)
    motion_signal_pub.publish("start")
    whole_body.move_end_effector_pose(geometry.pose(z=next_z, y=next_y, x=0.03), hole_frame_name)
    rospy.sleep(1.0)
    motion_signal_pub.publish("finish")

def move_robot_cb(msg):
    global step, next_z, next_y, object_weight, z_counter, y_counter, positive_dir, negative_dir
    global fail_flag, hole_id

    hole_frame_name = 'hole%d' %(hole_id)
    print msg.data
    print "collision"
    if not fail_flag:
        if msg.data < object_weight - 1.0:
            if z_counter > 5:
                z_counter = 0
                y_counter = y_counter + 1
            if positive_dir:
                next_z = init_z - step * z_counter
                next_y = init_y - step * y_counter

            if negative_dir:
                next_z = init_z + step * z_counter
                next_y = init_y + step * y_counter
            motion_signal_pub.publish("start")
            whole_body.move_end_effector_pose(geometry.pose(z=next_z, y=next_y, x=0.03), hole_frame_name)
            if y_counter > 5 and positive_dir == True:
                positive_dir = False
                negative_dir = True
                y_counter = 0
                print "negative_dir"
        
            if y_counter > 5 and negative_dir == True:
                fail_flag = True
                print "fail"
            
            print "motion fin"
            z_counter = z_counter + 1
            rospy.sleep(1.0)
            motion_signal_pub.publish("finish")

        if msg.data > object_weight - 1.0:
            print "found hole"
            whole_body.move_end_effector_pose(geometry.pose(z=next_z, y=next_y, x=-0.01), hole_frame_name)   
            gripper.command(1.0)
            rospy.sleep(1.0)
            whole_body.move_end_effector_pose(geometry.pose(z=next_z, y=next_y, x=0.2), hole_frame_name)
            hole_id = hole_id + 1
            print "next hole"
            hole_frame_name = 'hole%d' %(hole_id)
            print hole_frame_name
            if (hole_id < 4):
                whole_body.move_end_effector_pose(geometry.pose(x=0.2), hole_frame_name)
                motion_signal_pub.publish("next")

sub = rospy.Subscriber("ikea_force_feedback", Float32 , move_robot_cb , queue_size=1)
sub = rospy.Subscriber("ikea_object_weight", Float32 , object_weight_cb , queue_size=1)


rospy.spin()






        
