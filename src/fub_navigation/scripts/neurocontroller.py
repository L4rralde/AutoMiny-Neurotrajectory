#!/home/emmanuel/delfin/write/autominy/catkin_ws/src/fub_navigation/scripts/neuro/bin/python
import rospy
from std_msgs.msg import Int32MultiArray
from Controller_GA import NeuroEvolutionary
from autominy_msgs.msg import SteeringCommand, SpeedCommand
import numpy as np
import pygame 

#steering in degrees, vel in m/s
class NeuroEvolController:
    def __init__(self, max_steer=1, max_speed=0.5, model_name='model_1000_2'):
        pygame.init()
        rospy.init_node('NeuroEvolController')
        self.last_time = rospy.Time.now()

        self.agent = NeuroEvolutionary(max_speed*10, max_steer*180/np.pi)
        	
        self.agent.kinematic_ga.load_model(model_name) 
        	
        self.sub_dists = rospy.Subscriber('/rovislab/dists', Int32MultiArray, self.on_dist, queue_size=1)
        self.pub_speed = rospy.Publisher("/actuators/speed", SpeedCommand, queue_size=1, tcp_nodelay=True)
        rospy.on_shutdown(self.shutdown)
        self.shutdown_ = False
        self.pub_steer = rospy.Publisher("/actuators/steering", SteeringCommand,
                                       queue_size=1, tcp_nodelay=True)

    def on_dist(self, dists):
        dt = (rospy.Time.now()-self.last_time).to_sec()
        if dt < 0.04: 
            return 
        self.last_time = rospy.Time.now()

        steer_deg, vel_ps = self.agent.run_ga(self.agent.kinematic_ga.model, dists) ##???? dists, speed(?), time (?)
        self.steering = steer_deg*np.pi/180
        self.speed = vel_ps/10
        #Autominy's "/actuators/speed" is in m/s
        #gridsim car.velocity is in pixels/s = 0.1m/s
        #Autominy's "/actuators/steering" is in rad
        #gridsim car.steering is in degrees. 

        #n_rays = 31
        steerMsg = SteeringCommand()
        steerMsg.value = self.steering
        #steerMsg.value = 0.0
        steerMsg.header.frame_id = "base_link"
        steerMsg.header.stamp = rospy.Time.now()
        self.pub_steer.publish(steerMsg)

        if not self.shutdown_:
            msg = SpeedCommand()
            msg.value = self.speed
            #msg.value = 1.0
            msg.header.frame_id = "base_link"
            msg.header.stamp = rospy.Time.now()
            self.pub_speed.publish(msg)

    def shutdown(self):
        print("shutdown!")
        self.shutdown_ = True
        msg = SpeedCommand()
        msg.value = 0
        msg.header.frame_id = "base_link"
        self.pub_speed.publish(msg)
        rospy.sleep(1)
    
def main():
    print("Hello from neuroController")
   
    try:
        NeuroEvolController(0.7, 0.3, 'model_1000_2')
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("NeuroEvolController node terminated.")

if __name__ == '__main__':
    main()


            
