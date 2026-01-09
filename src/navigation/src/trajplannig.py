#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from my_teleop_joy.srv import SetGoal, ComputePath
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class TrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('trajectory_planner')
        
        # Clients pour appeler les services
        self.compute_path_client = self.create_client(ComputePath, '/computepath')
        
        # Service pour définir le goal
        self.set_goal_srv = self.create_service(
            SetGoal,
            '/Setgoal',
            self.set_goal_callback)
        
        # Service pour commencer la navigation
        self.start_nav_srv = self.create_service(
            Empty,
            '/start_navigation',
            self.start_navigation_callback)
        
        # Souscriptions
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        # Publisher pour les commandes
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # État
        self.goal_x = None
        self.goal_y = None
        self.pose = None
        self.path = []
        self.navigation_active = False
        self.path_computed = False
        
        # Timer pour la boucle de contrôle
        self.create_timer(0.5, self.control_loop)
        
        self.get_logger().info("TrajectoryPlanner initialized")

    def set_goal_callback(self, request, response):
        """Service: définir le goal"""
        self.goal_x = request.goal_x
        self.goal_y = request.goal_y
        self.path_computed = False
        self.path = []
        
        response.success = True
        response.message = f"Goal set to ({self.goal_x}, {self.goal_y})"
        
        self.get_logger().info(f"Goal set: ({self.goal_x}, {self.goal_y})")
        
        return response

    def start_navigation_callback(self, request, response):
        """Service: commencer la navigation"""
        
        if self.goal_x is None or self.goal_y is None:
            response.success = False
            response.message = "No goal set"
            return response
        
        if self.pose is None:
            response.success = False
            response.message = "Robot pose not available"
            return response
        
        # Appeler le service de calcul du chemin
        if not self.compute_path():
            response.success = False
            response.message = "Failed to compute path"
            return response
        
        self.navigation_active = True
        response.success = True
        response.message = "Navigation started"
        
        self.get_logger().info("✓ Navigation started")
        
        return response

    def compute_path(self):
        """Appeler le service ComputePath"""
        
        while not self.compute_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('compute_path service not available')
            return False
        
        request = ComputePath.Request()
        request.start_x = self.pose.position.x
        request.start_y = self.pose.position.y
        request.goal_x = self.goal_x
        request.goal_y = self.goal_y
        
        future = self.compute_path_client.call_async(request)
        future.add_done_callback(self.path_computed_callback)
        
        return True

    def path_computed_callback(self, future):
        """Callback quand le chemin est calculé"""
        try:
            response = future.result()
            if response.success:
                # Convertir les coords (stockées en cm) en mètres
                self.path = [(x / 100.0, y / 100.0) for x, y in 
                            zip(response.path_x, response.path_y)]
                self.path_computed = True
                self.get_logger().info(f"✓ Path received: {len(self.path)} waypoints")
            else:
                self.get_logger().error(f"Path computation failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def odom_callback(self, msg: Odometry):
        """Mettre à jour la pose du robot"""
        self.pose = msg.pose.pose

    def get_yaw(self, q):
        """Extraire yaw du quaternion"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def normalize_angle(self, angle):
        """Normaliser l'angle entre -π et +π"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def control_loop(self):
        """Boucle de contrôle principal"""
        
        if not self.navigation_active or not self.path_computed:
            return
        
        if self.pose is None or not self.path:
            return
        
        rx = self.pose.position.x
        ry = self.pose.position.y
        current_yaw = self.get_yaw(self.pose.orientation)
        
        # Waypoint suivant
        next_wp = self.path[0]
        wx, wy = next_wp[0], next_wp[1]
        
        # Calculs de distance et angle
        dx = wx - rx
        dy = wy - ry
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        err_yaw = self.normalize_angle(desired_yaw - current_yaw)
        
        # Contrôle
        k_rho = 2.0
        k_alpha = 2.5
        k_beta = -1.0
        
        linear_speed = k_rho * dist
        MAX_LINEAR_SPEED = 0.1
        linear_speed = max(-MAX_LINEAR_SPEED, min(MAX_LINEAR_SPEED, linear_speed))
        
        angular_speed = k_alpha * err_yaw + k_beta * err_yaw
        MAX_ANGULAR_SPEED = 0.5
        angular_speed = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angular_speed))
        
        # Publier
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_publisher.publish(twist)
        
        # Vérifier si waypoint atteint
        WAYPOINT_THRESHOLD = 0.15
        if dist < WAYPOINT_THRESHOLD:
            self.path.pop(0)
            remaining = len(self.path)
            if remaining > 0:
                self.get_logger().info(f"✓ Waypoint reached! {remaining} remaining")
            else:
                self.get_logger().info("🎉 GOAL REACHED!")
                self.navigation_active = False
                # Arrêter le robot
                twist = Twist()
                self.cmd_publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()