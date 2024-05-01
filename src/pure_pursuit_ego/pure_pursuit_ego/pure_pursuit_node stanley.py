#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from math import atan
from ackermann_msgs.msg import AckermannDriveStamped
# TODO CHECK: include needed ROS msg type headers and libraries
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

import pickle
import sys
sys.path.append('/home/shiming/Documents/ese615/race2/raceline_opt/dissertation-master/python')

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # TODO: create ROS subscribers and publishers
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_sub_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.waypoint_pub = self.create_publisher(PointStamped, '/waypoint', 10)

        # parameters
        # position_lookahead_time = 0.5
        cof = 1.5
        mass = 5.0
        self.steer_cap = 0.4
        self.ke = 8.0
        self.kv = 0.1

        # load the trajectory
        with open('/home/shiming/Documents/ese615/race2/raceline_opt/dissertation-master/data/plots/my_track/compromise/{}_{}.pickle'.format(mass, cof), 'rb') as f:
            trajectory = pickle.load(f)
        
        trajectory_position = trajectory.path.position(trajectory.s).T
        trajectory_heading = np.arctan2(np.diff(trajectory_position[:,1]), np.diff(trajectory_position[:,0]))
        target = np.vstack((trajectory_heading, trajectory.velocity.v)).T

        # remove the last point
        trajectory_position = trajectory_position[:-1]

        # upsample the trajectory position and target
        x = np.linspace(0, len(trajectory_position)-1, num=10000)

        f = interp1d(range(len(trajectory_position)), trajectory_position, axis=0)
        self.trajectory_position = f(x)
        self.neigh = NearestNeighbors(n_neighbors=1)
        self.neigh.fit(self.trajectory_position)

        f = interp1d(range(len(target)), target, axis=0)
        self.target = f(x)

        print("PurePursuit Initialized")


    def odom_sub_callback(self, odom_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        neigh_dist, neigh_ind = self.neigh.kneighbors(np.array([[odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]]))
        target = self.target[neigh_ind.squeeze()]
        nearest_point = self.trajectory_position[neigh_ind.squeeze()]

        # TODO: transform goal point to vehicle frame of reference
        arb = R.from_quat([
            odom_msg.pose.pose.orientation.x, 
            odom_msg.pose.pose.orientation.y, 
            odom_msg.pose.pose.orientation.z, 
            odom_msg.pose.pose.orientation.w])
        yaw = arb.as_euler('zyx')[0]

        # compute heading error
        heading_error = target[0]-yaw

        # convert to -pi to pi
        if heading_error > np.pi:
            heading_error -= 2*np.pi
        if heading_error < -np.pi:
            heading_error += 2*np.pi

        # compute cross track error
        cross_track_error = atan((self.ke*neigh_dist[0][0])/(self.kv+odom_msg.twist.twist.linear.x))
        cross_track_error *= np.sign(np.cross([np.cos(yaw), np.sin(yaw)], [nearest_point[0]-odom_msg.pose.pose.position.x, nearest_point[1]-odom_msg.pose.pose.position.y]))

        # compute steering angle
        steering_angle = heading_error+cross_track_error
        steering_angle = min(max(steering_angle, -self.steer_cap), self.steer_cap)

        speed = target[1]

        # TODO: publish drive message, don't forget to limit the steering angle.
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.get_logger().info('Publishing: steering_angle: %f, speed: %f' % (steering_angle, speed))
        self.drive_pub.publish(drive_msg)

        # publish waypoint
        waypoint_msg = PointStamped()
        waypoint_msg.point.x = nearest_point[0]
        waypoint_msg.point.y = nearest_point[1]
        waypoint_msg.header.frame_id = 'map'
        self.waypoint_pub.publish(waypoint_msg)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
