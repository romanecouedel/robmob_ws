#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import traceback
import numpy as np

from tf2_ros import TransformListener, Buffer


class DynamicObstacleAvoidance(Node):
    """
    Node pour l'évitement d'obstacles dynamique
    
    Combine :
    - Planification de chemin (waypoints)
    - Détection LIDAR (obstacles temps réel)
    - Contrôle réactif pour éviter les obstacles
    """
    
    def __init__(self):
        super().__init__('dynamic_obstacle_avoidance')

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.path_sub = self.create_subscription(
            Path,
            '/computed_path',
            self.path_callback,
            10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ✨ Variables de base
        self.scan = None
        self.iteration_count = 0
        self.path_found = False
        self.path = []
        self.path_computed = False
        self.navigation_active = False
        
        self.start_position = None
        self.returning_to_start = False
        self.goal_reached = False
        
        # ✨ Paramètres de contrôle
        self.MAX_LINEAR_SPEED = 0.5
        self.MAX_ANGULAR_SPEED = 1.0
        self.MIN_LINEAR_SPEED = 0.1
        self.WAYPOINT_THRESHOLD = 0.3
        
        self.k_rho = 1.5
        self.k_alpha = 2.5
        self.k_beta = -0.5
        
        self.velocity_filter_alpha = 0.3
        self.last_linear_speed = 0.0
        self.last_angular_speed = 0.0
        
        # ✨ PARAMÈTRES D'ÉVITEMENT D'OBSTACLES
        self.obstacle_threshold = 0.5      # Distance min détection obstacle (m)
        self.danger_zone = 1.0             # Zone dangereuse (m)
        self.critical_distance = 0.3       # Distance critique d'arrêt (m)
        self.obstacle_avoidance_enabled = True  # Activer/désactiver l'évitement
        
        # ✨ Secteurs LIDAR pour détection directionnelle
        self.front_angle_range = 30        # Angle avant en degrés (±)
        self.side_angle_range = 90         # Angle latéral en degrés
        
        # ✨ États d'évitement
        self.obstacle_detected = False
        self.obstacle_direction = None     # 'left', 'right', 'front'
        self.avoidance_mode = False        # Mode évitement actif
        
        self.get_logger().info("🤖 Dynamic Obstacle Avoidance initialized")
        self.get_logger().info(f"Obstacle threshold: {self.obstacle_threshold} m")
        self.get_logger().info(f"Danger zone: {self.danger_zone} m")
        self.create_timer(0.05, self.cmd)

    def scan_callback(self, msg: LaserScan):
        """Traiter les données LIDAR"""
        self.scan = msg
        self.analyze_obstacles()

    def analyze_obstacles(self):
        """
        ✨ Analyser les obstacles détectés par le LIDAR
        
        Divise le LIDAR en secteurs :
        - AVANT : danger immédiat
        - GAUCHE/DROITE : direction d'évitement
        """
        if self.scan is None:
            return
        
        ranges = self.scan.ranges
        angle_min = self.scan.angle_min
        angle_increment = self.scan.angle_increment
        
        # Initialiser les détections par secteur
        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')
        
        # Analyser chaque rayon du LIDAR
        for i, r in enumerate(ranges):
            # Ignorer les mesures invalides
            if math.isnan(r) or math.isinf(r) or r < self.scan.range_min or r > self.scan.range_max:
                continue
            
            # Calculer l'angle du rayon
            angle = angle_min + i * angle_increment
            angle_deg = math.degrees(angle)
            
            # Normaliser l'angle entre -180 et 180
            while angle_deg > 180:
                angle_deg -= 360
            while angle_deg < -180:
                angle_deg += 360
            
            # Classer le rayon par secteur
            if abs(angle_deg) < self.front_angle_range:
                # AVANT
                front_min = min(front_min, r)
            
            if -180 < angle_deg < -self.side_angle_range:
                # GAUCHE
                left_min = min(left_min, r)
            
            if self.side_angle_range < angle_deg < 180:
                # DROITE
                right_min = min(right_min, r)
        
        # Détecter les obstacles et déterminer la direction
        self.obstacle_detected = False
        self.obstacle_direction = None
        
        if front_min < self.danger_zone:
            self.obstacle_detected = True
            
            # Obstacle devant : chercher direction d'évitement
            if left_min > right_min:
                self.obstacle_direction = 'left'
            else:
                self.obstacle_direction = 'right'
            
            if self.iteration_count % 10 == 0:
                self.get_logger().info(
                    f"🚨 Obstacle détecté! Avant: {front_min:.2f}m, "
                    f"Gauche: {left_min:.2f}m, Droite: {right_min:.2f}m, "
                    f"Direction: {self.obstacle_direction}"
                )

    def path_callback(self, msg: Path):
        """Recevoir le chemin calculé"""
        if len(msg.poses) == 0:
            return

        self.path = [(pose.pose.position.x, pose.pose.position.y)
                     for pose in msg.poses]
        self.path_computed = True
        self.path_found = True
        self.navigation_active = True

        self.get_logger().info(f"✓ Chemin reçu: {len(self.path)} waypoints")

    def get_robot_pose(self):
        """Récupérer la pose du robot via TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                rclpy.time.Time()
            )
            
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            yaw = self.get_yaw(q)
            
            return (x, y, yaw)
            
        except Exception as e:
            self.get_logger().warn(f"TF indisponible: {e}")
            return None

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

    def publish_goal(self, x, y):
        """Publier un nouveau goal"""
        from geometry_msgs.msg import PoseStamped
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        goal_pub.publish(goal_msg)

    def low_pass_filter(self, current_value, last_value, alpha):
        """Filtre passe-bas"""
        return alpha * current_value + (1 - alpha) * last_value

    def compute_obstacle_avoidance(self, current_yaw):
        """
        ✨ Calculer les commandes d'évitement d'obstacle
        
        Stratégie :
        1. Si obstacle devant : ralentir et tourner
        2. Si côté libre : tourner vers ce côté
        3. Si très proche : arrêter et attendre
        """
        
        if not self.obstacle_detected:
            return None  # Pas d'obstacle
        
        # ✨ Cas 1 : Très proche -> ARRÊT D'URGENCE
        if self.scan is not None:
            min_range = min([r for r in self.scan.ranges 
                            if not (math.isnan(r) or math.isinf(r))])
            
            if min_range < self.critical_distance:
                self.get_logger().warn(f"🛑 ARRÊT D'URGENCE! Distance: {min_range:.2f}m")
                return (0.0, 0.0)  # Arrêter complètement
        
        # ✨ Cas 2 : Obstacle détecté -> Tourner pour éviter
        if self.obstacle_direction == 'left':
            # Tourner à gauche
            angular_speed = self.MAX_ANGULAR_SPEED * 0.8
        else:
            # Tourner à droite
            angular_speed = -self.MAX_ANGULAR_SPEED * 0.8
        
        # Réduire la vitesse linéaire pendant l'évitement
        linear_speed = self.MIN_LINEAR_SPEED * 0.5
        
        return (linear_speed, angular_speed)

    def control_law_improved(self, dx, dy, err_yaw, dist):
        """Loi de commande améliorée (suivi waypoint)"""
        
        if dist < 0.1:
            linear_speed = 0.0
            angular_speed = self.k_alpha * err_yaw
        
        elif abs(err_yaw) > math.pi / 3:
            linear_speed = 0.1 * math.cos(err_yaw)
            angular_speed = self.k_alpha * err_yaw + self.k_beta * err_yaw
        
        else:
            linear_speed = self.k_rho * dist
            angular_speed = self.k_alpha * err_yaw + self.k_beta * err_yaw
        
        return linear_speed, angular_speed

    def apply_speed_limits(self, linear_speed, angular_speed):
        """Appliquer les limites de vitesse"""
        
        if abs(angular_speed) > 0.5:
            max_linear = self.MAX_LINEAR_SPEED * (1.0 - abs(angular_speed) / self.MAX_ANGULAR_SPEED)
        else:
            max_linear = self.MAX_LINEAR_SPEED
        
        linear_speed = max(-max_linear, min(max_linear, linear_speed))
        angular_speed = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, angular_speed))
        
        if 0 < abs(linear_speed) < self.MIN_LINEAR_SPEED:
            linear_speed = math.copysign(self.MIN_LINEAR_SPEED, linear_speed)
        
        return linear_speed, angular_speed

    def apply_velocity_smoothing(self, linear_speed, angular_speed):
        """Lisser les commandes"""
        linear_speed = self.low_pass_filter(
            linear_speed, 
            self.last_linear_speed, 
            self.velocity_filter_alpha
        )
        angular_speed = self.low_pass_filter(
            angular_speed, 
            self.last_angular_speed, 
            self.velocity_filter_alpha
        )
        
        self.last_linear_speed = linear_speed
        self.last_angular_speed = angular_speed
        
        return linear_speed, angular_speed

    def cmd(self):
        """
        ✨ Boucle de contrôle principale
        
        Logique :
        1. Si obstacle détecté -> Mode évitement
        2. Sinon -> Suivre le chemin planifié
        """

        if not self.path_computed:
            return

        if not self.path_found:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_publisher.publish(twist)
            return

        if not self.navigation_active:
            return

        pose = self.get_robot_pose()
        if pose is None:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_publisher.publish(twist)
            return

        rx, ry, current_yaw = pose

        if self.start_position is None:
            self.start_position = (rx, ry)
            self.get_logger().info(f"📌 Position de départ: ({rx:.2f}, {ry:.2f})")

        # Gestion du cycle aller-retour
        if not self.path:
            if not self.returning_to_start:
                self.get_logger().info("✨ BUT ATTEINT!")
                self.goal_reached = True
                self.returning_to_start = True
                self.path_computed = False
                self.path_found = False
                
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_publisher.publish(twist)
                
                self.publish_goal(self.start_position[0], self.start_position[1])
                return
            else:
                self.get_logger().info("🎉 RETOUR RÉUSSI!")
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_publisher.publish(twist)
                
                self.navigation_active = False
                self.returning_to_start = False
                self.goal_reached = False
                return

        # Obtenir le prochain waypoint
        next_wp = self.path[0]
        wx, wy = next_wp[0], next_wp[1]

        dx = wx - rx
        dy = wy - ry
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        err_yaw = self._normalize_angle(desired_yaw - current_yaw)

        # ✨ LOGIQUE D'ÉVITEMENT D'OBSTACLES
        if self.obstacle_avoidance_enabled and self.obstacle_detected:
            # Mode évitement actif
            avoidance_cmd = self.compute_obstacle_avoidance(current_yaw)
            
            if avoidance_cmd is not None:
                linear_speed, angular_speed = avoidance_cmd
                self.avoidance_mode = True
                
                if self.iteration_count % 10 == 0:
                    self.get_logger().info(
                        f"[ÉVITEMENT] Robot: ({rx:.2f}, {ry:.2f}) | "
                        f"Direction: {self.obstacle_direction} | "
                        f"v={linear_speed:.2f}, ω={angular_speed:.2f}"
                    )
        else:
            # Mode normal : suivre le waypoint
            self.avoidance_mode = False
            linear_speed, angular_speed = self.control_law_improved(dx, dy, err_yaw, dist)
            
            if self.iteration_count % 20 == 0:
                self.get_logger().info(
                    f"[SUIVI] Robot: ({rx:.2f}, {ry:.2f}) | "
                    f"Waypoint: ({wx:.2f}, {wy:.2f}) | "
                    f"Dist: {dist:.3f}"
                )

        # Appliquer les limites et lissage
        linear_speed, angular_speed = self.apply_speed_limits(linear_speed, angular_speed)
        linear_speed, angular_speed = self.apply_velocity_smoothing(linear_speed, angular_speed)

        # Publier les commandes
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_publisher.publish(twist)

        # Vérifier si waypoint atteint (sauf en mode évitement)
        if not self.avoidance_mode and dist < self.WAYPOINT_THRESHOLD:
            self.path.pop(0)
            remaining = len(self.path)
            if remaining > 0:
                self.get_logger().info(f"✓ Waypoint atteint! {remaining} restants")

        self.iteration_count += 1


def main(args=None):
    rclpy.init(args=args)
    try:
        navigator = DynamicObstacleAvoidance()
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