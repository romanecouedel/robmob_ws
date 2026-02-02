#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np

class ExplorationNode(Node):
    """
    Nœud d'exploration ultra-simple.
    
    Algorithme:
    1. Avancer tout droit
    2. Si obstacle devant, tourner
    3. Recommencer
    """
    
    def __init__(self):
        super().__init__('exploration_node')
        
        # État
        self.exploration_enabled = False
        self.obstacle_ahead = False
        
        # Vitesses
        self.forward_speed = 0.15  # m/s
        self.turn_speed = 0.4      # rad/s
        self.obstacle_distance = 0.6  # m
        
        # Subscribers
        self.enable_sub = self.create_subscription(
            Bool,
            '/exploration/enable',
            self.enable_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer de contrôle (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Exploration Node initialisé)')
    
    def enable_callback(self, msg):
        """Activer/désactiver l'exploration"""
        self.exploration_enabled = msg.data
        
        if msg.data:
            self.get_logger().info('EXPLORATION ACTIVÉE')
        else:
            self.get_logger().info('EXPLORATION DÉSACTIVÉE')
            # Arrêter le robot
            stop = Twist()
            self.cmd_pub.publish(stop)
    
    def scan_callback(self, msg):
        """Vérifier s'il y a un obstacle devant"""
        if not self.exploration_enabled:
            return
        
        # Récupérer les distances + filtrage
        ranges = np.array(msg.ranges)
        ranges = ranges[~np.isnan(ranges)]  # Enlever les NaN
        ranges = ranges[ranges > 0]  # Enlever les 0
        
        if len(ranges) == 0:
            return
        
        # Prendre seulement le secteur avant
        num_points = len(ranges)
        angle_front = int(num_points * 0.15)  # 15% de chaque côté = 30° environ (15% de 180°)
        
        # Secteur avant
        front_left = ranges[:angle_front]
        front_right = ranges[-angle_front:]
        front_ranges = np.concatenate([front_left, front_right])
        
        # Obstacle si la distance min est inférieure au seuil
        if len(front_ranges) > 0:
            min_distance = np.min(front_ranges)
            self.obstacle_ahead = min_distance < self.obstacle_distance
    
    def control_loop(self):
        """Boucle de contrôle principale"""
        if not self.exploration_enabled:
            return
        
        twist = Twist()
        
        # Algorithme simple :
        if self.obstacle_ahead:
            # TOURNER
            twist.angular.z = self.turn_speed
            self.get_logger().info('Obstacle détecté - Rotation', throttle_duration_sec=2.0)
        else:
            # AVANCER
            twist.linear.x = self.forward_speed
        
        # Publier la commande
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Arrêt')
    finally:
        # Arrêter le robot
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()