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

move_to_global_init_flag = False
hand_over_flag = False
natural_weight = 0
natural_weight_counter = 0
get_init_weight_flag = False
finish_init_flag = False

hole_id = 0

def global_init_pose():
    global move_to_global_init_flag
    whole_body.move_to_neutral()
    whole_body.move_to_joint_positions({'arm_lift_joint': 0.4})
    whole_body.move_to_joint_positions({'arm_flex_joint': -1})
    whole_body.move_to_joint_positions({'wrist_flex_joint': -0.5})
    whole_body.move_to_joint_positions({'head_pan_joint': -1.0})
    whole_body.move_to_joint_positions({'head_tilt_joint': -0.8})
    gripper.command(1.0)
    rospy.sleep(2.0)
    move_to_global_init_flag = True

def move_robot_cb(msg):
    global init_flag, init_move_flag,  force_fb_buffer, init_counter
    global object_weight, motion_finish_flag, motion_start_flag
    global hand_over_flag,natural_weight,natural_weight_counter, get_init_weight_flag,move_to_global_init_flag, hole_id

    print msg.data
    if msg.data == "start":
        print "motion_start"
        rospy.sleep(0.5)
        motion_start_flag = True
    if msg.data == "finish":
        print "motion_finish"
        motion_start_flag = False
        motion_finish_flag = True
    if msg.data == "next":
        print "next hole"
        hole_id = hole_id + 1        
        init_flag = False
        init_move_flag = False
        object_weight = 0
        init_counter = 0
        motion_finish_flag = False
        motion_start_flag = False
        force_fb_buffer = []
        hand_over_flag = False
        natural_weight = 0
        natural_weight_counter = 0
        get_init_weight_flag = False
        finish_init_flag = False


def force_cb(msg):    
    global init_flag, init_move_flag,  force_fb_buffer, init_counter
    global object_weight, motion_finish_flag, motion_start_flag
    global hand_over_flag,natural_weight,natural_weight_counter, get_init_weight_flag,move_to_global_init_flag, hole_id

    hole_frame_name = 'hole%d' %(hole_id)
    if move_to_global_init_flag:
        # wait for hand over
        if not hand_over_flag:
            if not get_init_weight_flag:
                if natural_weight_counter < 5:
                    natural_weight = natural_weight + msg.wrench.force.x
                    natural_weight_counter = natural_weight_counter + 1
                else:
                    natural_weight = natural_weight/natural_weight_counter
                    print "natural_weight"
                    print natural_weight
                    get_init_weight_flag = True
            else:
                if msg.wrench.force.x < (natural_weight-2.0) or msg.wrench.force.x > (natural_weight+2.0):
                    rospy.sleep(3)
                    print "touched"
                    gripper.command(-0.6)
                    rospy.sleep(3)
                    hand_over_flag = True
                    return

        else:
            if not init_flag:
                if not init_move_flag:
                    whole_body.move_end_effector_pose(geometry.pose(x=0.05), hole_frame_name)
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
                    #print "????????"
                    print len(force_fb_buffer)
                    print min(force_fb_buffer)
                    force_pub.publish(min(force_fb_buffer))
                    del force_fb_buffer[:]
                    print len(force_fb_buffer)
                    motion_finish_flag = False

sub = rospy.Subscriber("/hsrb/wrist_wrench/raw", WrenchStamped , force_cb, queue_size=1)
sub = rospy.Subscriber("ikea_motion_signal", String , move_robot_cb , queue_size=1)
global_init_pose()
rospy.spin()






        
