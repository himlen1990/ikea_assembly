from hsrb_interface import Robot
import rospy
from hsrb_interface import geometry
from geometry_msgs.msg import WrenchStamped, Wrench
from std_msgs.msg import String



natural_weight = 0
init_counter = 0
cal_init_weight_flag = False
finish_init_flag = False

robot = Robot()
whole_body = robot.try_get('whole_body')
gripper = robot.try_get('gripper')
whole_body.move_to_neutral()
whole_body.move_to_joint_positions({'arm_lift_joint': 0.4})
whole_body.move_to_joint_positions({'arm_flex_joint': -1})
whole_body.move_to_joint_positions({'wrist_flex_joint': -0.5})
whole_body.move_to_joint_positions({'head_pan_joint': -1.0})
whole_body.move_to_joint_positions({'head_tilt_joint': -0.8})
whole_body.end_effector_frame = u'hand_palm_link'
whole_body.linear_weight = 100.0
whole_body.impedance_config = 'compliance_middle'


def force_cb(msg):
    global natural_weight,init_counter, cal_init_weight_flag, finish_init_flag

    if not cal_init_weight_flag:
        if init_counter < 5:
            natural_weight = natural_weight + msg.wrench.force.x
            init_counter = init_counter + 1
            print natural_weight
            print init_counter
        elif init_counter >= 5:
            natural_weight = natural_weight/init_counter
            print "natural_weight" 
            print natural_weight
            cal_init_weight_flag = True

    if  cal_init_weight_flag and not finish_init_flag:       
        if msg.wrench.force.x < (natural_weight-2.0) or msg.wrench.force.x > (natural_weight+2.0):
            finish_init_flag = True
            rospy.sleep(3)            
            print "touched"
            print msg.wrench.force.x
            gripper.command(-0.7)


sub = rospy.Subscriber("/hsrb/wrist_wrench/raw", WrenchStamped , force_cb, queue_size=1)

rospy.spin()






        
