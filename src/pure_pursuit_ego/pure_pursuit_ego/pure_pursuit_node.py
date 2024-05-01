
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from math import atan
from ackermann_msgs.msg import AckermannDriveStamped
# TODO CHECK: include needed ROS msg type headers and libraries
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from visualization_msgs.msg import Marker

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_dilation
from skimage.segmentation import flood

import matplotlib.pyplot as plt

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # TODO: create ROS subscribers and publishers
        qos = 1
        self.odom_sub = self.create_subscription(Odometry, '/pf/pose/odom', self.odom_sub_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_sub_callback, qos)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', qos)
        # self.waypoint_pub = self.create_publisher(PointStamped, '/waypoint_ego', qos)
        self.marker_pub = self.create_publisher(Marker, '/occupancy_grid', qos)


        self.map_client = self.create_client(GetMap, "/map_server/map")
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        request = GetMap.Request()
        future = self.map_client.call_async(request)
        future.add_done_callback(self.map_callback)

        # flags
        self.map_init = True
        self.pose_init = True
        self.scan_init = True

        # parameters
        self.position_lookahead_time = 0.5
        self.lookahead_speed_min = 2.0
        self.steer_cap = 0.5
        self.steer_gain = 1.0
        self.margin = 0.6
        self.margin_index = None
        self.caution_distance = 2.0
        self.stop_distance = 1.0
        self.caution_distance_min = 0.2
        self.lateral_caution_distance = 0.4
        self.lateral_stop_distance = 0.2
        self.caution_angle = np.pi/2
        self.eps = 1e-3

        lv = 2.0
        mv = 3.2
        hv = 5.8

        # waypoints
        waypoint = np.array([
            [7.4, 10.7, lv],
            [6.4, 9.9, mv],
            [5.2, 6.8, hv],
            [2.9, 4.8, mv],
            [2.4, 2.5, lv],
            [3.9, 1.2, mv],
            [5.8, 1.0, hv],
            [8.7, 3.2, mv],
            [9.2, 5.5, hv],
            [8.9, 9.7, mv],
            [7.4, 10.7, lv]
        ])

        # fit splines
        tck, _ = splprep(waypoint.T, s=0, per=True)
        dense_waypoints = np.array(splev(np.linspace(0, 1, 100), tck))

        # # plot the waypoints
        # plt.figure()
        # plt.scatter(dense_waypoints[0], dense_waypoints[1], c=dense_waypoints[2])
        # plt.axis('equal')
        # plt.show()

        s = np.cumsum(np.linalg.norm(np.diff(dense_waypoints[:2, :]), axis=0))
        s = np.insert(s, 0, 0)

        self.trajectory_position = dense_waypoints[:2, :].T
        self.trajectory_cumsum = np.cumsum(np.diff(s))
        self.trajectory_cumsum = np.insert(self.trajectory_cumsum, 0, 0)

        self.target = dense_waypoints[2, :-1]
        self.kn_regressor = NearestNeighbors()
        self.kn_regressor.fit(self.trajectory_position[:-1])

        print("PurePursuit Initialized")


    def map_callback(self, future):
        response = future.result()

        # Set the origin of the occupancy grid
        self.r_map_world = np.array(
            [response.map.info.origin.position.x, 
                response.map.info.origin.position.y])
        
        # Set the resolution of the occupancy grid
        self.resolution = response.map.info.resolution

        # Set margin index
        self.margin_index = int(self.margin / self.resolution)

        # Set the flag to indicate that the occupancy grid has been received
        self.map_init = False

        # Convert the occupancy grid to a numpy array
        self.static_map = np.array(response.map.data).reshape(
            (response.map.info.height, response.map.info.width)).astype(bool)
        binary_dilation(
            self.static_map, 
            iterations=self.margin_index, 
            output=self.static_map)
        
        # get the seed point, transform it to the map frame
        seed_point_in_world_frame = np.array([1.0, 2.0])
        seed_point_in_map_frame = np.round((seed_point_in_world_frame - self.r_map_world) / self.resolution).astype(int)

        # update the static map using skimage.segmentation.flood
        self.static_map = flood(self.static_map, tuple(np.flip(seed_point_in_map_frame)))
        
        # # visualize the occupancy grid
        # marker = Marker()
        # marker.header.frame_id = "map"
        # marker.header.stamp = self.get_clock().now().to_msg()
        # marker.ns = "occupancy_grid"
        # marker.id = 0
        # marker.type = Marker.CUBE_LIST
        # marker.action = Marker.ADD
        # marker.pose.orientation.w = 1.0
        # marker.scale.x = self.resolution
        # marker.scale.y = self.resolution
        # marker.scale.z = 0.1
        # marker.color.a = 1.0
        # for i in range(self.static_map.shape[0]):
        #     for j in range(self.static_map.shape[1]):
        #         if self.static_map[i, j]:
        #             point = Point()
        #             point.x = j * self.resolution + self.r_map_world[0]
        #             point.y = i * self.resolution + self.r_map_world[1]
        #             marker.points.append(point)
        # self.marker_pub.publish(marker)


        # visualize the trajectory
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.points = [Point(x=point[0], y=point[1], z=0.0) for point in self.trajectory_position]
        self.marker_pub.publish(marker)


    def odom_sub_callback(self, odom_msg):
        if self.pose_init:
            self.pose_init = False
        self.curr_position = np.array([
            odom_msg.pose.pose.position.x, 
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z])
        self.curr_orientation = R.from_quat([
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w])
        self.curr_speed = odom_msg.twist.twist.linear.x


    def scan_sub_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        # status check
        if self.map_init:
            return
        if self.pose_init:
            return
        if self.scan_init:
            theta = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
            self.cos_theta = np.cos(theta)
            self.sin_theta = np.sin(theta)
            self.zero_ranges = np.zeros(len(scan_msg.ranges))
            self.scan_init = False
        
        # update occupancy grid
        obs_in_lidar_frame = np.vstack([
            scan_msg.ranges * self.cos_theta, 
            scan_msg.ranges * self.sin_theta,
            self.zero_ranges])
        obs_in_lidar_frame = obs_in_lidar_frame[:, obs_in_lidar_frame[0, :]>=0]
        obs_in_world_frame = self.curr_position + self.curr_orientation.apply(obs_in_lidar_frame.T)
        obs_in_map_frame = np.round((obs_in_world_frame[:, :2] - self.r_map_world) / self.resolution).astype(int)

        # filter out the points outside the map
        keep_index = (obs_in_map_frame[:, 0] >= 0) & (obs_in_map_frame[:, 0] < self.static_map.shape[1]) & (obs_in_map_frame[:, 1] >= 0) & (obs_in_map_frame[:, 1] < self.static_map.shape[0])
        obs_in_map_frame = obs_in_map_frame[keep_index]
        obs_in_world_frame = obs_in_world_frame[keep_index]

        # filter out the points that are occupied in the static map
        obs_in_world_frame = obs_in_world_frame[self.static_map[obs_in_map_frame[:, 1], obs_in_map_frame[:, 0]]]

        # # plot the obs_in_world_frame
        # marker = Marker()
        # marker.header.frame_id = "map"
        # marker.type = marker.POINTS
        # marker.action = marker.ADD
        # marker.scale.x = 0.1
        # marker.scale.y = 0.1
        # marker.color.a = 1.0
        # marker.color.r = 1.0
        # marker.points = [Point(x=point[0], y=point[1], z=0.0) for point in obs_in_world_frame]
        # self.marker_pub.publish(marker)

        # TODO: find the current waypoint to track using methods mentioned in lecture
        current_position = self.curr_position[0:2]
        neigh_ind = self.kn_regressor.kneighbors(current_position.reshape(1, -1))[1][0][0]

        # get the speed
        caution_gain = self.get_caution_gain(obs_in_world_frame)
        speed = caution_gain*self.target[neigh_ind]

        # get lookahead_position
        trajectory_cumsum = self.trajectory_cumsum/max(self.lookahead_speed_min, self.curr_speed)
        t = trajectory_cumsum[neigh_ind]
        target_ind = np.searchsorted(trajectory_cumsum, (t + self.position_lookahead_time)%trajectory_cumsum[-1])
        lookahead_position = self.trajectory_position[target_ind]

        # TODO: transform goal point to vehicle frame of reference
        lookahead_position_in_map_frame = np.insert(lookahead_position-current_position, 2, 0)
        lookahead_position_in_vehicle_frame = self.curr_orientation.inv().apply(lookahead_position_in_map_frame)

        # TODO: calculate curvature/steering angle and speed
        L = np.linalg.norm(lookahead_position_in_vehicle_frame)
        gamma = lookahead_position_in_vehicle_frame[1]/(L**2)
        steering_angle = atan(self.steer_gain*gamma)
        # steering_angle = self.steer_gain * atan2(lookahead_position_in_vehicle_frame[1], lookahead_position_in_vehicle_frame[0])
        steering_angle = min(max(steering_angle, -self.steer_cap), self.steer_cap)

        # TODO: publish drive message, don't forget to limit the steering angle.
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.get_logger().info('Publishing: steering_angle: %f, speed: %f' % (steering_angle, speed))
        self.drive_pub.publish(drive_msg)

        # # publish the waypoint
        # waypoint_msg = PointStamped()
        # waypoint_msg.point.x = lookahead_position[0]
        # waypoint_msg.point.y = lookahead_position[1]
        # waypoint_msg.header.frame_id = "map"
        # self.waypoint_pub.publish(waypoint_msg)


    def get_caution_gain(self, obs_in_world_frame):
        # get the distance and angle between the opponent and the ego
        diff_obs_ego = obs_in_world_frame[:, :2] - self.curr_position[:2]
        distance_from_ego = np.linalg.norm(diff_obs_ego, axis=1)
        
        # keep the points that are within the caution distance
        keep_index = (distance_from_ego < self.caution_distance)&(distance_from_ego > self.caution_distance_min)
        obs_in_world_frame = obs_in_world_frame[keep_index]
        distance_from_ego = distance_from_ego[keep_index]

        # return 1 if there is no opponent
        if len(obs_in_world_frame) == 0:
            return 1.0

        # get the mean of the obs_in_world_frame and the distance from ego
        mean_obs_in_world_frame = np.mean(obs_in_world_frame, axis=0)
        mean_distance_from_ego = np.mean(distance_from_ego)

        # get the angle of the mean in the vehicle frame
        mean_obs_in_map_frame = np.insert(mean_obs_in_world_frame[:2]-self.curr_position[:2], 2, 0)
        mean_obs_in_vehicle_frame = self.curr_orientation.inv().apply(mean_obs_in_map_frame)
        angle = atan(mean_obs_in_vehicle_frame[1]/mean_obs_in_vehicle_frame[0])
        angle = min(max(angle, -self.caution_angle), self.caution_angle)

        # compute the caution gain
        caution_gain_from_ego = max(mean_distance_from_ego-self.stop_distance, 0)/(self.caution_distance-self.stop_distance)
        caution_gain_from_ego += self.eps
        caution_gain_from_ego **= (self.caution_angle - abs(angle)) / self.caution_angle
        if caution_gain_from_ego < 0.1:
            caution_gain_from_ego = 0.0

        return caution_gain_from_ego


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
