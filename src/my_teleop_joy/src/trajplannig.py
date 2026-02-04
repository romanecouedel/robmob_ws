#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import qos_profile_sensor_data

from tf2_ros import TransformListener, Buffer
import math
import traceback


class TrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('trajectory_planner')

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publisher pour envoyer le goal (retour au départ)
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)

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

        self.scan = None
        self.iteration_count = 0
        self.path_found = False
        self.path = []
        self.path_computed = False
        self.navigation_active = False
        
        # Nouvelles variables pour gérer le cycle retour
        self.start_position = None  # Position de départ (sauvegardée)
        self.returning_to_start = False  # État: en train de revenir au départ
        self.goal_reached = False  # Le but a-t-il été atteint ?
        
        # NOUVELLE: Phase de réorientation
        self.orienting = False  # Est-on en train de se réorienter ?
        self.ORIENTATION_THRESHOLD = 0.1  # Tolérance en radians (~5.7°)
        
        self.get_logger().info("TrajectoryPlanner initialized - waiting for /computed_path and TF map->base_footprint")
        self.create_timer(0.01, self.cmd)

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
        self.orienting = True  # Commencer par la phase de réorientation
        
        #  Si on revient au départ, on réinitialise l'état
        if self.returning_to_start:
            self.get_logger().info("✓ Nouveau chemin reçu pour retour au départ")
        else:
            self.get_logger().info(f"✓ Chemin reçu: {len(self.path)} waypoints")

    def get_robot_pose(self):
        """
        Récupérer la pose du robot via TF map->base_footprint
        Retourne: (x, y, yaw) ou None si TF indisponible
        """
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
        """
        Publier un nouveau goal pour que path_manager recalcule le chemin
        """
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f"Goal publié: ({x:.2f}, {y:.2f})")

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

        # Récupérer la position actuelle via TF
        pose = self.get_robot_pose()
        if pose is None:
            # TF indisponible, arrêter le robot
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_publisher.publish(twist)
            return

        rx, ry, current_yaw = pose

        #  Sauvegarder la position de départ (première itération)
        if self.start_position is None:
            self.start_position = (rx, ry)
            self.get_logger().info(f" Position de départ sauvegardée: ({rx:.2f}, {ry:.2f})")

        # Vérifier s'il y a des waypoints restants
        if not self.path:
            #  si le goal est atteint on leve un flag qui dit oui le but est atteint mais on doit revenir au point de depart et la on baisse la flag de le calcule de path et on recalcule avec un nouveau goal qui est le point de depart
            if not self.returning_to_start:
                # Le but a été atteint
                self.get_logger().info(" BUT ATTEINT! ")
                self.get_logger().info(f" Position actuelle: ({rx:.2f}, {ry:.2f})")
                self.goal_reached = True
                self.returning_to_start = True
                self.path_computed = False
                self.path_found = False
                
                # Publier un nouveau goal = point de départ
                self.publish_goal(self.start_position[0], self.start_position[1])
                
                # Arrêter temporairement le robot
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_publisher.publish(twist)
                return
            else:
                # On est revenu au point de départ !
                self.get_logger().info(" RETOUR AU POINT DE DÉPART RÉUSSI! ")
                self.get_logger().info(f" Position finale: ({rx:.2f}, {ry:.2f})")
                
                # Arrêter le robot
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_publisher.publish(twist)
                
                # Réinitialiser pour un nouveau cycle
                self.navigation_active = False
                self.returning_to_start = False
                self.goal_reached = False
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

        # ============ PHASE 1 : RÉORIENTATION SUR PLACE ============
        if self.orienting:
            # Vérifier si on est bien orienté
            if abs(err_yaw) < self.ORIENTATION_THRESHOLD:
                # On est orienté ! Passer à la phase de suivi
                self.orienting = False
                self.get_logger().info("✓ Orientation OK ! Passage au suivi du chemin")
            else:
                # Tourner sur place pour s'orienter
                angular_speed = 1.5 * err_yaw  # Gain simple pour la rotation
                MAX_ANGULAR_SPEED = 0.8
                angular_speed = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angular_speed))
                
                twist = Twist()
                twist.linear.x = 0.0  # PAS DE DÉPLACEMENT LINÉAIRE
                twist.angular.z = angular_speed  # SEULEMENT LA ROTATION
                self.cmd_publisher.publish(twist)
                
                if self.iteration_count % 20 == 0:
                    self.get_logger().info(f"[Orientation] Err: {err_yaw:.3f} rad | Angular speed: {angular_speed:.3f}")
                
                self.iteration_count += 1
                return  # Important: ne pas continuer jusqu'au suivi

        # ============ PHASE 2 : SUIVI DU CHEMIN ============
        #  Débogage
        if self.iteration_count % 10 == 0:
            mode = "Retour" if self.returning_to_start else "Aller"
            self.get_logger().info(
                f"[{mode}] Robot: ({rx:.2f}, {ry:.2f}, yaw={current_yaw:.2f}) | "
                f"Waypoint: ({wx:.2f}, {wy:.2f}) | "
                f"Dist: {dist:.3f}"
            )

        # Contrôle proportionnel
        k_rho = 2.0
        k_alpha = 0.8  # Réduit pour éviter les grands mouvements
        k_beta = -1.0

        linear_speed = k_rho * dist
        MAX_LINEAR_SPEED = 0.3
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
                mode = "Retour" if self.returning_to_start else "Aller"
                self.get_logger().info(f"✓ [{mode}] Waypoint atteint! {remaining} restants")
                self.orienting = True  # Réorienter avant le prochain waypoint

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