#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import qos_profile_sensor_data

from tf2_ros import TransformListener, Buffer
import math
import traceback


class TrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('trajectory_planner')

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.path_sub = self.create_subscription(
            Path,
            '/computed_path',
            self.path_callback,
            10)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.pose_callback,
            qos_profile_sensor_data
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialiser avec une pose par défaut
        pose_test = PoseWithCovarianceStamped()
        pose_test.pose.pose.position.x = 0.0
        pose_test.pose.pose.position.y = 0.0
        pose_test.pose.pose.position.z = 0.0
        
        self.pose = pose_test.pose.pose  # Stocker la pose complète
        self.orientation = pose_test.pose.pose.orientation
        
        self.scan = None
        self.iteration_count = 0
        self.path_found = False
        self.path = []
        self.path_computed = False
        self.navigation_active = False
        
        self.get_logger().info("TrajectoryPlanner initialized - waiting for /amcl_pose and /computed_path")
        self.create_timer(0.5, self.cmd)

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.pose = msg.pose.pose
        self.orientation = msg.pose.pose.orientation

    def scan_callback(self, msg):
        self.scan = msg

    def path_callback(self, msg: Path):
        """Recevoir le chemin calculé"""
        if len(msg.poses) == 0:
            return

        # Convertir le chemin en liste de tuples (x, y)
        self.path = [(pose.pose.position.x, pose.pose.position.y)
                     for pose in msg.poses]
        self.path_computed = True
        self.path_found = True
        self.navigation_active = True

        self.get_logger().info(f"✓ Chemin reçu: {len(self.path)} waypoints")

    def get_yaw(self, q):
        """Extraire yaw d'un quaternion"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _normalize_angle(self, angle):
        """Normaliser un angle entre -π et +π"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def cmd(self):
        """Contrôle du robot pour suivre le chemin planifié"""

        # Attendre le chemin
        if not self.path_computed:
            return

        # Si pas de chemin trouvé, arrêter
        if not self.path_found:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_publisher.publish(twist)
            return

        # Si navigation pas active
        if not self.navigation_active:
            return

        # Récupérer la position actuelle
        rx = self.pose.position.x
        ry = self.pose.position.y
        current_yaw = self.get_yaw(self.orientation)

        # Vérifier s'il y a des waypoints restants
        if not self.path:
            self.get_logger().info("BUT ATTEINT!")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_publisher.publish(twist)
            self.navigation_active = False
            return

        # Obtenir le prochain waypoint
        next_wp = self.path[0]
        wx, wy = next_wp[0], next_wp[1]

        # Calculer distance et angle
        dx = wx - rx
        dy = wy - ry
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        err_yaw = self._normalize_angle(desired_yaw - current_yaw)

        # 🔍 Débogage
        if self.iteration_count % 10 == 0:
            self.get_logger().info(
                f"Robot: ({rx:.2f}, {ry:.2f}) | "
                f"Waypoint: ({wx:.2f}, {wy:.2f}) | "
                f"Dist: {dist:.3f}"
            )

        # Contrôle proportionnel
        k_rho = 2.0
        k_alpha = 2.5
        k_beta = -1.0

        linear_speed = k_rho * dist
        MAX_LINEAR_SPEED = 0.1
        linear_speed = max(-MAX_LINEAR_SPEED, min(MAX_LINEAR_SPEED, linear_speed))

        angular_speed = k_alpha * err_yaw + k_beta * err_yaw
        MAX_ANGULAR_SPEED = 0.5
        angular_speed = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angular_speed))

        # Publier les commandes
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_publisher.publish(twist)

        # Vérifier si waypoint atteint
        WAYPOINT_THRESHOLD = 0.3
        if dist < WAYPOINT_THRESHOLD:
            self.path.pop(0)
            remaining = len(self.path)
            if remaining > 0:
                self.get_logger().info(f"✓ Waypoint atteint! {remaining} restants")
            else:
                self.get_logger().info("BUT ATTEINT!")

        self.iteration_count += 1


def main(args=None):
    rclpy.init(args=args)
    try:
        navigator = TrajectoryPlanner()
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        print("Arrêt par utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()