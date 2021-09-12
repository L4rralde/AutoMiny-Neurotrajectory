#!/usr/bin/env python2
from autominy_msgs.msg import SteeringCommand, SpeedCommand
import numpy as np
import cv2 
from cv_bridge import CvBridge, CvBridgeError
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion

class Visualizer: 
    def __init__(self):
        rospy.init_node('visualize')
        self.cv_bridge = CvBridge()

        self.pose = [0,0,0] 
        rospy.on_shutdown(self.shutdown)
        self.shutdown_ = False

        self.speed = 0
        self.steer = 0
        self.pub_vector = rospy.Publisher("/rovislab/wanted_vector", Image, queue_size=1)
        self.sub_odom = rospy.Subscriber("/sensors/localization/filtered_map", Odometry, self.on_odometry, queue_size=1)
        self.sub_steering = rospy.Subscriber("/actuators/steering", SteeringCommand, self.on_steercmd, queue_size=1, tcp_nodelay=True)
        self.sub_speed = rospy.Subscriber("/actuators/speed", SpeedCommand, self.on_speedcmd, queue_size=1, tcp_nodelay=True)
        self.sub_grid = rospy.Subscriber("/rovislab/grid", Image, self.on_grid, queue_size=1) 


    def on_odometry(self, data): 
        x = data.pose.pose.position.x*100.0
        y = data.pose.pose.position.y*100.0

        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.pose = [int(x),int(y),yaw]

    def on_steercmd(self, data): 
        self.steer = data.value

    def on_speedcmd(self, data): 
        self.speed = data.value

    def on_grid(self, grid): 
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(grid, "bgr8")
        except CvBridgeError as e:
            print(e)
        height, width, channels = cv_image.shape        
        vec_len = int(min(height,width)*self.speed/2)
        ang = self.pose[2]-self.steer+np.pi
        cv2.line(cv_image, (int(width/2), int(height/2)), 
            (int(width/2-vec_len*np.cos(ang)),int(height/2-vec_len*np.sin(ang))),
            (0,0,255), 2)
        try: 
            self.pub_vector.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def shutdown(self):
        print("shutdown!")
        self.shutdown_ = True
        rospy.sleep(1)
        
def main():
    try:
        Visualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Visualizer node terminated")

if __name__ == '__main__':
    main()
