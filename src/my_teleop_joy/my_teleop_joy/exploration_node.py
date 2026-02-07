#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
import cv2
import time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


class ExplorationNode(Node):

    def __init__(self):
        super().__init__('exploration_node')

        # -------------------- ÉTAT --------------------
        self.exploration_enabled = False
        self.exploration_mode = 'random'   # random | frontier | done
        self.waiting_for_goal_completion = False
        self.obstacle_ahead = False

        # -------------------- PARAMÈTRES --------------------
        self.forward_speed = 0.15
        self.turn_speed = 0.4
        self.obstacle_distance = 0.4
        self.exploration_threshold = 40.0
        self.min_frontier_distance = 0.5

        # -------------------- MAP --------------------
        self.map_data = None
        self.map_info = None
        self.exploration_percentage = 0.0
        self.threshold_reached = False
        self.last_map_time = None

        # -------------------- FRONTIERS --------------------
        self.frontier_points = []
        self.current_frontier_index = -1

        # -------------------- SÉCURITÉ MAP --------------------
        self.forced_random_until = None
        self.saved_mode_before_forced_random = None

        # -------------------- TF --------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------------------- VISU --------------------
        self.show_visualization = True
        cv2.namedWindow('Frontier Exploration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frontier Exploration', 800, 800)

        # -------------------- QOS --------------------
        qos_map = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # -------------------- SUBS --------------------
        self.create_subscription(Bool, '/exploration/enable', self.enable_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_map)
        self.create_subscription(Bool, '/goal_reached', self.goal_reached_callback, 10)

        # -------------------- PUBS --------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # -------------------- TIMERS --------------------
        self.create_timer(0.01, self.control_loop)
        self.create_timer(0.5, self.update_visualization)

        self.get_logger().info("🚀 Exploration Node prêt")


    # ======================================================
    # LOG UTILS
    # ======================================================

    def log_state(self):
        self.get_logger().debug(
            f"[STATE] mode={self.exploration_mode} | "
            f"waiting={self.waiting_for_goal_completion} | "
            f"frontiers={len(self.frontier_points)} | "
            f"explored={self.exploration_percentage:.1f}%"
        )

    # ======================================================
    # CALLBACKS
    # ======================================================

    def enable_callback(self, msg):
        self.exploration_enabled = msg.data

        if msg.data:
            self.get_logger().info("🟢 Exploration ACTIVÉE")
            self.exploration_mode = 'random'
            self.threshold_reached = False
            self.waiting_for_goal_completion = False
            self.last_map_time = time.time()
        else:
            self.get_logger().warn("🔴 Exploration DÉSACTIVÉE")
            self.cmd_pub.publish(Twist())

        self.log_state()

    def map_callback(self, msg):
        if not self.exploration_enabled:
            return

        self.last_map_time = time.time()
        self.map_info = msg.info
        self.map_data = np.array(msg.data)

        total = len(self.map_data)
        explored = np.sum(self.map_data != -1)
        self.exploration_percentage = (explored / total) * 100.0

        self.get_logger().info(
            f"📊 Exploration: {self.exploration_percentage:.1f}%",
            throttle_duration_sec=5.0
        )

        # ---- Transition random -> frontier ----
        if not self.threshold_reached and self.exploration_percentage >= self.exploration_threshold:
            self.threshold_reached = True
            self.exploration_mode = 'frontier'
            self.get_logger().info("🚀 TRANSITION random → frontier")

            self.find_frontier_points()
            self.go_to_next_frontier()

        if self.exploration_mode == 'frontier':
            self.find_frontier_points()

            if self.waiting_for_goal_completion and not self.is_frontier_still_valid():
                self.get_logger().warn("❌ Frontière invalide → suivante")
                self.waiting_for_goal_completion = False
                self.go_to_next_frontier()

    def goal_reached_callback(self, msg):
        if not msg.data:
            return

        if not self.waiting_for_goal_completion:
            self.get_logger().debug("Goal déjà traité (ignoré)")
            return

        self.waiting_for_goal_completion = False
        self.get_logger().info(
            f"✅ GOAL ATTEINT index={self.current_frontier_index}"
        )

        self.find_frontier_points()
        self.go_to_next_frontier()
        self.log_state()

    def scan_callback(self, msg):
        if not self.exploration_enabled or self.exploration_mode != 'random':
            return

        ranges = np.array(msg.ranges)
        ranges = ranges[np.isfinite(ranges)]
        ranges = ranges[ranges > 0]

        if len(ranges) == 0:
            return

        front = int(len(ranges) * 0.15)
        front_ranges = np.concatenate([ranges[:front], ranges[-front:]])
        self.obstacle_ahead = np.min(front_ranges) < self.obstacle_distance

    # ======================================================
    # FRONTIERS
    # ======================================================

    def find_frontier_points(self):
        h = self.map_info.height
        w = self.map_info.width
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y

        grid = self.map_data.reshape((h, w))
        self.frontier_points = []

        for y in range(3, h-3):
            for x in range(3, w-3):
                if 0 <= grid[y, x] < 50:
                    neighbors = [grid[y-1,x], grid[y+1,x], grid[y,x-1], grid[y,x+1]]
                    if any(n == -1 for n in neighbors):
                        wx = ox + (x+0.5)*res
                        wy = oy + (y+0.5)*res
                        self.frontier_points.append({'x': wx, 'y': wy, 'gx': x, 'gy': y})

        self.get_logger().info(
            f"🔍 {len(self.frontier_points)} frontières détectées",
            throttle_duration_sec=3.0
        )

    def go_to_next_frontier(self):
        if self.waiting_for_goal_completion:
            return

        if len(self.frontier_points) == 0:
            self.get_logger().info("🏁 PLUS DE FRONTIÈRES → DONE")
            self.exploration_mode = 'done'
            return

        pose = self.get_robot_pose()
        if not pose:
            self.get_logger().warn("TF robot indisponible")
            return

        rx, ry = pose
        best_i = None
        best_d = float('inf')

        for i, p in enumerate(self.frontier_points):
            d = math.hypot(p['x']-rx, p['y']-ry)
            if d > self.min_frontier_distance and d < best_d:
                best_d = d
                best_i = i

        if best_i is None:
            self.get_logger().info("🏁 PLUS DE FRONTIÈRES VALIDES → DONE")
            self.exploration_mode = 'done'
            return

        self.current_frontier_index = best_i
        target = self.frontier_points[best_i]

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = target['x']
        goal.pose.position.y = target['y']
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)
        self.waiting_for_goal_completion = True

        self.get_logger().info(
            f"🎯 NOUVEAU GOAL index={best_i} dist={best_d:.2f}m"
        )

    # ======================================================
    # UTILS
    # ======================================================

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def is_frontier_still_valid(self):
        if self.current_frontier_index < 0 or self.current_frontier_index >= len(self.frontier_points):
            return False

        p = self.frontier_points[self.current_frontier_index]
        grid = self.map_data.reshape((self.map_info.height, self.map_info.width))
        return 0 <= grid[p['gy'], p['gx']] < 50

    # ======================================================
    # CONTROL LOOP
    # ======================================================

    def control_loop(self):
        if not self.exploration_enabled:
            return

        twist = Twist()

        if self.exploration_mode == 'random':
            if self.obstacle_ahead:
                twist.angular.z = self.turn_speed
            else:
                twist.linear.x = self.forward_speed
            self.cmd_pub.publish(twist)

        elif self.exploration_mode == 'done':
            self.cmd_pub.publish(Twist())

    # ======================================================
    # VISU
    # ======================================================

    def update_visualization(self):
        if not self.show_visualization or self.map_data is None:
            return

        h = self.map_info.height
        w = self.map_info.width
        grid = self.map_data.reshape((h, w))
        img = np.zeros((h, w, 3), dtype=np.uint8)

        img[grid == -1] = [50,50,50]
        img[(grid >= 0) & (grid < 50)] = [255,255,255]
        img[grid >= 50] = [0,0,0]

        for i,p in enumerate(self.frontier_points):
            c = (0,255,0) if i == self.current_frontier_index else (255,255,0)
            cv2.circle(img, (p['gx'], p['gy']), 3, c, -1)

        cv2.imshow("Frontier Exploration", img)
        cv2.waitKey(1)


# ======================================================
# MAIN
# ======================================================

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt utilisateur")
    finally:
        node.cmd_pub.publish(Twist())
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
