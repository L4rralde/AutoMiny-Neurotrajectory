#!/usr/bin/env python2
import numpy as np
import cv2 
from cv_bridge import CvBridge, CvBridgeError
import math
import rospkg
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray
from tf.transformations import euler_from_quaternion
import tf

class GridDrawer: 
    def __init__(self, ray_len=20, ray_n=100, poles_d=11, lidar_d=17, dx=0.0187, dy=-0.329, th=0.3905):
        rospy.init_node('MakeGrid')
        self.cv_bridge = CvBridge()
        self.listener = tf.TransformListener()
        rospack = rospkg.RosPack()

        self.ray_len = ray_len
        self.ray_n = ray_n
        self.poles_d = poles_d
        self.lidar_d = lidar_d
        self.map_size_x = 596 #cm
        self.map_size_y = 596 #cm 
        self.file_path = rospack.get_path('make_grid') + '/scripts/'
        self.map_img = cv2.imread(self.file_path+'map.bmp') 
        #self.map_img = cv2.flip(self.map_img, 1)
        self.lut = cv2.imread(self.file_path + 'lut.bmp')
        self.pose = [0,0,0] 
        self.dx = dx
        self.dy = dy
        self.th = 0.3905

        rospy.on_shutdown(self.shutdown)
        self.shutdown_ = False
        self.pub_grid = rospy.Publisher("/rovislab/grid", Image, queue_size=1)
        self.pub_dists = rospy.Publisher("/rovislab/dists", Int32MultiArray, queue_size=1)
        #self.pub_pose = rospy.Publisher("/rovislab/pose", Int32MultiArray, queue_size=1)
        #self.pub_distances py.Publisher("/rovislab/distances",Type ???, queue_size=1)
        self.sub_odom = rospy.Subscriber("/sensors/localization/filtered_map", Odometry, self.on_odometry, queue_size=1)
        self.sub_lidar = rospy.Subscriber("/sensors/rplidar/scan", LaserScan, self.on_lidar, queue_size=1) 


    def on_odometry(self, data): 
        x = data.pose.pose.position.x*100.0
        y = data.pose.pose.position.y*100.0

        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        #self.pose = [int(x),int(y),yaw]
        self.pose = [int(x),int(y),yaw]
        #pose_msg = Int32MultiArray()
        #pose_msg.data = self.pose
        #dists_msg.header.stamp = rospy.Time.now() ???
        #self.pub_pose.publish(pose_msg)

    def on_lidar(self, data):         
        points = []
        #Getting points from lidar's frame and transforming them to car frame
        for i in range(0, int(len(data.ranges)*(1.0/6.0))-1, 1): 
            d = 100*data.ranges[i]
            sigma = i*data.angle_increment+np.pi/2
            if np.abs(d)<5*self.ray_len:
                points.append([int(d),sigma])
        for i in range(len(data.ranges)-1, int(len(data.ranges)*(1-1.0/6.0)),-1):
            d = 100*data.ranges[i]
            sigma = i*data.angle_increment-3*np.pi/2
            if np.abs(d)<5*self.ray_len:
                points.append([int(d),sigma])

        #Sampling from car frame points and mask
        points = np.asarray(points)
        #points = points[~np.isnan(points)]

        cob = int(self.ray_len*1.25)

        x = self.pose[0]
        y = self.pose[1]

        min_x_roi = max(0, x-cob)
        min_y_roi = max(0, y-cob)
        max_x_roi = min(self.map_size_x, x+cob)
        max_y_roi = min(self.map_size_y, y+cob)

        map_roi = self.map_img[min_y_roi:max_y_roi,min_x_roi:max_x_roi].copy()
        rect = np.zeros((2*cob,2*cob), np.uint8)
        rect = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
        #rect[y-min_y_roi:max_y_roi-y,x-min_x_roi:max_x_roi-x] = map_roi      
        rect_x1 = cob+min_x_roi-x
        rect_x2 = cob+max_x_roi-x
        rect_y1 = cob-y+min_y_roi
        rect_y2 = cob-y+max_y_roi
        #print(rect_x1, rect_x2, rect_y1, rect_y2)
        #print(min_x_roi, max_x_roi, min_y_roi, max_y_roi)
        rect[rect_y1:rect_y2,rect_x1:rect_x2] = map_roi
        roi_car_x = rect_x1+int((-min_x_roi+max_x_roi)/2)
        roi_car_y = rect_y1+int((-min_y_roi+max_y_roi)/2)

        #cob = int(cob*scale)
        #map_roi = cv2.resize(map_roi, (2*cob,2*cob))
        dists = []
        for s in np.linspace(np.pi/6, np.pi*5/6, self.ray_n): 
            i = self.findNearest(points,s) #points array includes NaNs
            d_obs = points[i,0]
            map_ang = s+self.pose[2]+np.pi/2
            maskRayDist = self.getMaskDistance(self.pose, map_ang)
            dist = min(maskRayDist,d_obs)
            dists.append(dist)
            cv2.line(rect,(cob,cob),(int(cob-dist*np.cos(map_ang)),int(cob-dist*np.sin(map_ang))),(0,255,0), 1)
        print(np.asarray(dists))
        #dists_msg = gsray()
        dists_msg = Int32MultiArray()
        dists_msg.data = dists
        #dists_msg.header.stamp = rospy.Time.now() ???
        self.pub_dists.publish(dists_msg)
        try: 
            self.pub_grid.publish(self.cv_bridge.cv2_to_imgmsg(rect, "bgr8"))
        except CvBridgeError as e:
            print(e)


    def transformedPoint(self, index, msg): 
        return 0

    def findNearest(self, array, value): 
        #array = np.asarray(array)
        array = array[:,-1]
        idx = (np.abs(array - value)).argmin()
        return idx

    def getMaskDistance(self, pose, angle): 
        x = pose[0]
        y = pose[1]
        mask = False
        step = 0
        while (not mask) and (step<self.ray_len): 
            x_r = int(x-step*np.cos(angle))
            y_r = int(y-step*np.sin(angle))
            mask = (self.lut[x_r,y_r,0]==0)
            step = step+1
        return step

    def shutdown(self):
        print("shutdown!")
        self.shutdown_ = True
        rospy.sleep(1)


"""
class GridDrawer:
    def __init__(self,l1=200,l2=150):
        rospy.init_node('MakeGrid')
        #self.map_size_x = 600  # cm
        #selff.map_size_y = 430  # cm
        self.l1 = l2
        self.l2 = l1
        #self.grid = np.ones((self.l1, self.l2, 3), dtype=int)
        self.roi = np.array([[0,0],[0,self.l2],[self.l1, self.l2],[self.l1,0]])
        self.cv_bridge = CvBridge()
        self.map_size_x = 596  # cm
        self.map_size_y = 596  # cm
        self.resolution = 1  # cm
        self.listener = tf.TransformListener()
        rospack = rospkg.RosPack()
        self.file_path = rospack.get_path('make_grid') + '/src/'
        self.map_img = cv2.imread(self.file_path+'map.bmp') 
        self.lut = cv2.imread(self.file_path + 'lut.bmp')
        self.matrix_lane_1 = np.load(self.file_path + 'matrix50cm_lane1.npy')
        self.matrix_lane_2 = np.load(self.file_path + 'matrix50cm_lane2.npy')
        self.distance_lane_1 = np.load(self.file_path + 'matrix0cm_lane1.npy')
        self.distance_lane_2 = np.load(self.file_path + 'matrix0cm_lane2.npy')
    
        rospy.on_shutdown(self.shutdown)
        self.shutdown_ = False
        self.pub_road = rospy.Publisher("/sensors/road", Image, queue_size=1)
        self.pub_obs = rospy.Publisher("/sensors/obstacles", Image, queue_size=1)
        self.pub_hough = rospy.Publisher("/sensors/hough", Image, queue_size=1)
        self.sub_odom = rospy.Subscriber("/sensors/localization/filtered_map", Odometry, self.callback, queue_size=1)
        self.sub_lidar = rospy.Subscriber("/sensors/rplidar/scan", LaserScan, self.on_lidar, queue_size=1) 
        #self.poles_sub = 
        self.max_dist=(0.25*self.l1**2+0.25*self.l2**2)**0.5
"""

"""
    def callback(self, data):
        #print("Hello from GridDrawer callback")
        x = data.pose.pose.position.x*100.0
        y = data.pose.pose.position.y*100.0

        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        self.roi = self.getRoi(x,y,yaw)
"""

"""
    def getRoi(self, x, y, yaw): 
        s = np.array([x,y])
        c_yaw = np.cos(yaw-np.pi/2)
        s_yaw = np.sin(yaw-np.pi/2)
        mid = np.array([-s_yaw, c_yaw])
        edg = np.array([c_yaw, s_yaw])
        ps1 = s+0.5*self.l2*mid
        ps2 = -(ps1-2*s)
        kedg = 0.5*self.l1*edg
        p1 = ps1-kedg
        p2 = ps2-kedg
        p4 = ps1+kedg
        p3 = ps2+kedg
        p1[0] = min(596,max(0,p1[0]))
        p1[1] = min(596,max(0,p1[1]))
        p2[0] = min(596,max(0,p2[0]))
        p2[1] = min(596,max(0,p2[1]))
        p3[0] = min(596,max(0,p3[0]))
        p3[1] = min(596,max(0,p3[1]))
        p4[0] = min(596,max(0,p4[0]))
        p4[1] = min(596,max(0,p4[1]))
        roi = np.array([p1,p2,p3,p4], dtype=int)
        return roi
"""

"""
    def on_lidar(self, msg, min_angle=-np.pi, max_angle=np.pi):
        aff_frame= np.float32([[298-self.l1/2,298-self.l2/2],[298-self.l1/2,298+self.l2/2],[298+self.l1/2,298+self.l2/2]])
        roi_3 = self.roi[0:3,:].astype(np.float32)
        M_affine = cv2.getAffineTransform(roi_3, aff_frame)
        img_wAffd = cv2.warpAffine(self.map_img,M_affine,(596,596))
        lut_wAffd = cv2.warpAffine(self.lut,M_affine,(596,596)) 

        wAffd = img_wAffd[int(298-self.l1/2):int(298+self.l1/2+1),int(298-self.l2/2):int(298+self.l2/2+1)]
        #self.grid = lut_wAffd[int(298-self.l2/2):int(298+self.l2/2),int(298-self.l1/2):int(298+self.l1/2)]
        grid = lut_wAffd[int(298-self.l1/2):int(298+self.l1/2+1),int(298-self.l2/2):int(298+self.l2/2+1)]
        
        try: 
            self.pub_road.publish(self.cv_bridge.cv2_to_imgmsg(wAffd, "bgr8"))
        except CvBridgeError as e:
            print(e)
        points = []
        #poles = []
        for i in range(int(len(msg.ranges)*(1-(-min_angle/np.pi))),int(len(msg.ranges)*(max_angle/np.pi)-1), 1): 
            dist = -100*msg.ranges[i]
            angle = i*msg.angle_increment
            if np.abs(dist) < self.max_dist: 
            #if (np.abs(dist)<self.max_dist and (int(angle*10)<26 or int(angle*10)>38 or np.abs(dist)>22)):
                #print(angle, dist) 
                points.append((dist*np.cos(angle),dist*np.sin(angle)))
        #lines = cv2.HoughLines(points)
        roi_poitns = grid[:]
        #Transformada de Hough
        points_img = np.zeros((self.l1, self.l2), np.uint8)
        if len(points)>0: 
            for (x,y) in points: 
                x = int(self.l1/2+x)
                y = int(self.l2/2+y)
                if (x>=0 and x<self.l1) and (y>=0 and y<self.l2):
                    #print(x,y)
                    roi_poitns[x,y] = (0,0,0)
                    points_img[x,y] = 255
        try: 
            self.pub_obs.publish(self.cv_bridge.cv2_to_imgmsg(roi_poitns, "bgr8"))
        except CvBridgeError as e:
            print(e)

        lines = cv2.HoughLinesP(points_img, 1, np.pi/180, 2, None, 2, 5)
        roi_poitns_h = grid[:]
        if lines is not None: 
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(roi_poitns_h, (l[0],l[1]), (l[2],l[3]), (0,0,0), 1, cv2.LINE_AA)
        
        try: 
            self.pub_hough  .publish(self.cv_bridge.cv2_to_imgmsg(roi_poitns_h, "bgr8"))
        except CvBridgeError as e:
            print(e)


        #Lo siguiente tal vez ayudaria a mejorar la ubicacion de la nube de puntos. 
        
        points_on_track = 0

        if len(points) > 0:
            (t,r) = self.listener.lookupTransform("map", msg.header.frame_id, rospy.Time(0))
            mat44 = np.dot(tf.transformations.translation_matrix(t), tf.transformations.quaternion_matrix(r))

            for (x, y) in points:
                (xm, ym, zm) = tuple(np.dot(mat44, np.array([x, y, 0, 1.0])))[:3]
                (xi, yi) = int(xm * (100 / self.resolution)), int(ym * (100 / self.resolution))
                #if self.lut[xi,yi]:
        pass
""" 
"""    
    def shutdown(self):
        print("shutdown!")
        self.shutdown_ = True
        rospy.sleep(1)
"""       
        
def main():
    try:
        #n_rays = 30 (crei que eran 31...)
        #rays_len = 199 (crei que era 200)
        #conversion: Las escalas son iguales. 1 pixel
        #equivale a 1cm de un carro de tamanioo real
        GridDrawer(199, 30) 
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("GridDrawer node terminated")

if __name__ == '__main__':
    main()
