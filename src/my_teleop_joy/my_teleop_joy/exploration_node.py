#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
import numpy as np
import math

class ExplorationNode(Node):
    """
    Nœud d'exploration intelligent.
    
    Algorithme:
    1. Exploration aléatoire jusqu'à 50%
    2. À 50%, générer des points frontières
    3. Envoyer les frontières au PathManager (qui calcule le chemin)
    4. TrajectoryPlanner suit automatiquement le chemin
    5. Mettre à jour la liste quand un point est atteint
    """
    
    def __init__(self):
        super().__init__('exploration_node')
        
        # État
        self.exploration_enabled = False
        self.obstacle_ahead = False
        
        # Modes d'exploration
        self.exploration_mode = 'random'  # 'random' ou 'frontier'
        
        # Vitesses (mode random uniquement)
        self.forward_speed = 0.15  # m/s
        self.turn_speed = 0.4      # rad/s
        self.obstacle_distance = 0.4  # m
        
        # Suivi de l'exploration
        self.exploration_percentage = 0.0
        self.map_data = None
        self.map_info = None
        self.threshold_reached = False
        self.exploration_threshold = 50.0  # Seuil de 50%
        
        # Points frontières
        self.frontier_points = []
        self.current_frontier_index = 0
        
        # Position du robot
        self.robot_x = None
        self.robot_y = None
        
        # Tolérance pour atteindre un point frontière
        self.frontier_tolerance = 5  # m
        
        # Navigation state
        self.waiting_for_goal_completion = False
        
        # TF pour position du robot
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
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
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        # Subscriber pour savoir quand le chemin est terminé
        self.path_sub = self.create_subscription(
            Path,
            '/computed_path',
            self.path_callback,
            10
        )
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.nav_enable_pub = self.create_publisher(Bool, '/nav/enable', 10)
        
        # Timer de contrôle (10 Hz)
        self.timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info('Exploration Node initialisé')
        self.get_logger().info('Utilise PathManager + TrajectoryPlanner pour la navigation')
    
    def enable_callback(self, msg):
        """Activer/désactiver l'exploration"""
        self.exploration_enabled = msg.data
        
        if msg.data:
            self.get_logger().info('EXPLORATION ACTIVÉE')
            self.threshold_reached = False
            self.frontier_points = []
            self.exploration_mode = 'random'
            self.waiting_for_goal_completion = False
        else:
            self.get_logger().info('EXPLORATION DÉSACTIVÉE')
            # Arrêter le robot
            stop = Twist()
            self.cmd_pub.publish(stop)
            # Désactiver la navigation
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_enable_pub.publish(nav_msg)
    
    def map_callback(self, msg):
        """Calculer le pourcentage d'exploration de la carte"""
        if not self.exploration_enabled:
            return
        
        # Sauvegarder les infos de la carte
        self.map_info = msg.info
        self.map_data = np.array(msg.data)
        
        # Récupérer les données de la carte
        map_array = self.map_data
        
        total_cells = len(map_array)
        explored_cells = np.sum(map_array != -1)
        
        # Calculer le pourcentage
        if total_cells > 0:
            self.exploration_percentage = (explored_cells / total_cells) * 100.0
            
            # Afficher le pourcentage régulièrement
            self.get_logger().info(
                f'Exploration: {self.exploration_percentage:.2f}%',
                throttle_duration_sec=5.0
            )
            
            # Vérifier si le seuil est atteint
            if not self.threshold_reached and self.exploration_percentage >= self.exploration_threshold:
                self.threshold_reached = True
                self.get_logger().info(
                    f' SEUIL ATTEINT! La carte a été explorée à {self.exploration_percentage:.2f}% '
                    f'(>= {self.exploration_threshold}%)'
                )
                
                # Générer la liste des points frontières
                self.find_frontier_points(map_array, msg.info)
                
                # Passer en mode frontier
                self.exploration_mode = 'frontier'
                
                # Arrêter l'exploration aléatoire
                stop = Twist()
                self.cmd_pub.publish(stop)
                
                # Activer PathManager
                nav_msg = Bool()
                nav_msg.data = True
                self.nav_enable_pub.publish(nav_msg)
                
                # Aller au premier point frontière
                self.go_to_next_frontier()
            
            # Mettre à jour les frontières si déjà en mode frontier
            elif self.threshold_reached and self.exploration_mode == 'frontier':
                # Recalculer les frontières régulièrement
                old_count = len(self.frontier_points)
                self.find_frontier_points(map_array, msg.info, silent=True)
                new_count = len(self.frontier_points)
                
                if new_count != old_count:
                    self.get_logger().info(f'Frontières mises à jour: {new_count} points (était {old_count})')
                    
                    # Si plus de frontières
                    if new_count == 0:
                        self.get_logger().info(' EXPLORATION TERMINÉE - Plus de frontières!')
                        self.exploration_mode = 'done'
                        nav_msg = Bool()
                        nav_msg.data = False
                        self.nav_enable_pub.publish(nav_msg)
                        
    def is_frontier_safe(self, grid, x, y, safety_radius=2):
        """Vérifier qu'il n'y a pas d'obstacle proche"""
        height, width = grid.shape
        
        for dy in range(-safety_radius, safety_radius + 1):
            for dx in range(-safety_radius, safety_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny, nx] > 50:  # Obstacle
                        return False
        return True
    
    def find_frontier_points(self, map_array, map_info, silent=False):
        """
        Trouver les points frontières selon la règle suivante:
        - On considère les pixels "gris" (valeurs intermédiaires, ex 50-99) OU inconnus (-1)
        - Un pixel gris est une frontière valide s'il a au moins un voisin blanc (libre)
          et aucun voisin noir (occupé).

        Cette méthode évite de considérer comme frontière les voisins de gris qui sont
        à côté d'obstacles.
        """
        width = map_info.width
        height = map_info.height
        resolution = map_info.resolution
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y

        # Convertir en grille 2D
        grid = map_array.reshape((height, width))

        self.frontier_points = []

        # Parcourir la grille (éviter bords)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                val = grid[y, x]

                # Définir quand on considère une cellule comme "gris/inconnue"
                is_gray = (val == -1) or (50 <= val < 100)
                if not is_gray:
                    continue

                # Récupérer voisins (8-connexité)
                neighbors = [
                    grid[y-1, x], grid[y+1, x], grid[y, x-1], grid[y, x+1],
                    grid[y-1, x-1], grid[y-1, x+1], grid[y+1, x-1], grid[y+1, x+1],
                ]

                # Conditions: au moins un voisin blanc/libre, aucun voisin noir/occupé
                has_white_neighbor = any((n >= 0 and n < 50) for n in neighbors)
                has_black_neighbor = any(n >= 100 for n in neighbors)

                if has_white_neighbor and not has_black_neighbor:
                    world_x = origin_x + (x + 0.5) * resolution
                    world_y = origin_y + (y + 0.5) * resolution
                    self.frontier_points.append({
                        'x': world_x,
                        'y': world_y,
                        'grid_x': x,
                        'grid_y': y
                    })
        
        if not silent:
            self.get_logger().info(f'  {len(self.frontier_points)} points frontières trouvés!')
            
            if len(self.frontier_points) > 0:
                self.get_logger().info('Exemples de points frontières:')
                for i, point in enumerate(self.frontier_points[:5]):
                    self.get_logger().info(
                        f'  Point {i+1}: x={point["x"]:.2f}m, y={point["y"]:.2f}m'
                    )
                
                if len(self.frontier_points) > 5:
                    self.get_logger().info(f'  ... et {len(self.frontier_points) - 5} autres points')
                    
                
    def get_robot_pose(self):
        """Obtenir la position du robot via TF"""
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            return x, y
        except Exception as e:
            return None
    
    def go_to_next_frontier(self):
        """Envoyer un goal vers le prochain point frontière"""
        if len(self.frontier_points) == 0:
            self.get_logger().info('Plus de points frontières - Exploration terminée!')
            self.exploration_mode = 'done'
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_enable_pub.publish(nav_msg)
            return
        
        # Obtenir position robot
        pose = self.get_robot_pose()
        if pose:
            self.robot_x, self.robot_y = pose
        
        # Trouver le point frontière le plus proche
        if self.robot_x is not None and self.robot_y is not None:
            closest_idx = 0
            min_dist = float('inf')
            
            for i, point in enumerate(self.frontier_points):
                dist = math.sqrt((point['x'] - self.robot_x)**2 + (point['y'] - self.robot_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            self.current_frontier_index = closest_idx
        else:
            # Si pas de position, prendre le premier
            self.current_frontier_index = 0
        
        # Publier le goal
        target = self.frontier_points[self.current_frontier_index]
        
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = target['x']
        goal_msg.pose.position.y = target['y']
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal_msg)
        self.waiting_for_goal_completion = True
        
        self.get_logger().info(
            f'Navigation vers frontière {self.current_frontier_index + 1}/{len(self.frontier_points)}: '
            f'({target["x"]:.2f}, {target["y"]:.2f})'
        )
    
    def path_callback(self, msg):
        """
        Recevoir le chemin calculé
        (TrajectoryPlanner s'en occupe automatiquement)
        """
        if not self.exploration_enabled or self.exploration_mode != 'frontier':
            return
        
        if len(msg.poses) > 0:
            self.get_logger().info(f'✓ Chemin reçu: {len(msg.poses)} waypoints (TrajectoryPlanner s\'en charge)')
    
    def is_frontier_reached(self):
        """Vérifier si le point frontière actuel est atteint"""
        if len(self.frontier_points) == 0:
            return False
        
        pose = self.get_robot_pose()
        if pose is None:
            return False
        
        self.robot_x, self.robot_y = pose
        
        target = self.frontier_points[self.current_frontier_index]
        dx = target['x'] - self.robot_x
        dy = target['y'] - self.robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        return distance < self.frontier_tolerance
    
    def scan_callback(self, msg):
        """Vérifier s'il y a un obstacle devant (pour mode random)"""
        if not self.exploration_enabled or self.exploration_mode != 'random':
            return
        
        ranges = np.array(msg.ranges)
        ranges = ranges[~np.isnan(ranges)]
        ranges = ranges[ranges > 0]
        
        if len(ranges) == 0:
            return
        
        num_points = len(ranges)
        angle_front = int(num_points * 0.20)

        #angle_front = int(num_points * 0.70)
        #front_left = ranges[:-angle_front] # pour la vraie vie 
        front_left = ranges[:angle_front]

        front_right = ranges[-angle_front:]
        #front_ranges = np.concatenate([front_left])
        front_ranges = np.concatenate([front_left, front_right])
        
        if len(front_ranges) > 0:
            min_distance = np.min(front_ranges)
            self.obstacle_ahead = min_distance < self.obstacle_distance
    
    def control_loop(self):
        """Boucle de contrôle principale"""
        if not self.exploration_enabled:
            return
        
        twist = Twist()
        
        # Mode exploration aléatoire (avant 70%)
        if self.exploration_mode == 'random':
            if self.obstacle_ahead:
                twist.angular.z = self.turn_speed
                self.get_logger().info('Obstacle - Rotation', throttle_duration_sec=1.5)
            else:
                twist.linear.x = self.forward_speed
            
            self.cmd_pub.publish(twist)
        
        # Mode navigation vers frontières (après 50%)
        elif self.exploration_mode == 'frontier':
            
            # Vérifier si on attend la fin d'un trajet
            if self.waiting_for_goal_completion:
                
                # Vérifier si on a atteint la frontière
                if self.is_frontier_reached():
                    self.get_logger().info(f'✓ Frontière {self.current_frontier_index + 1} atteinte!')
                    
                    # Supprimer ce point de la liste
                    del self.frontier_points[self.current_frontier_index]
                    self.get_logger().info(f' {len(self.frontier_points)} frontières restantes')
                    
                    # Marquer comme complété
                    self.waiting_for_goal_completion = False
                    
                    # Rafraîchir la liste des frontières avec la nouvelle carte
                    if self.map_info is not None and self.map_data is not None:
                        self.get_logger().info(' Rafraîchissement de la liste des frontières...')
                        self.find_frontier_points(self.map_data, self.map_info, silent=True)
                    
                    # Petit délai avant le prochain
                    self.create_timer(1.0, self.go_to_next_frontier, single_shot=True)
            
            # TrajectoryPlanner gère le mouvement, on ne publie rien ici
        
        # Mode terminé
        elif self.exploration_mode == 'done':
            # Arrêter le robot
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Arrêt')
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
