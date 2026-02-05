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
import cv2

class ExplorationNode(Node):
    """
    Nœud d'exploration intelligent avec visualisation CV2.
    
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
        self.exploration_threshold = 40.0  # Seuil de 50%
        
        # Points frontières
        self.frontier_points = []
        self.current_frontier_index = 0
        
        # Position du robot
        self.robot_x = None
        self.robot_y = None

        # Navigation state
        self.waiting_for_goal_completion = False
        
        # Callback du goal atteint (sera rempli par goal_reached_callback)
        self.last_goal_reached_time = None
        
        # TF pour position du robot
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Visualisation CV2
        self.show_visualization = True
        cv2.namedWindow('Frontier Exploration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frontier Exploration', 800, 800)
        
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
        
        # Subscriber pour savoir quand le goal est atteint (depuis TrajectoryPlanner)
        self.goal_reached_sub = self.create_subscription(
            Bool,
            '/goal_reached',
            self.goal_reached_callback,
            10
        )
        
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.nav_enable_pub = self.create_publisher(Bool, '/nav/enable', 10)
        
        # Timer de contrôle (10 Hz)
        self.timer = self.create_timer(0.01, self.control_loop)
        
        # Timer pour visualisation CV2
        self.viz_timer = self.create_timer(0.5, self.update_visualization)
        
        self.get_logger().info('Exploration Node initialisé')
        self.get_logger().info('Utilise PathManager + TrajectoryPlanner pour la navigation')
        self.get_logger().info('Visualisation CV2 activée')
    
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
        if not self.exploration_enabled:
            return

        self.map_info = msg.info
        self.map_data = np.array(msg.data)

        map_array = self.map_data
        total_cells = len(map_array)
        explored_cells = np.sum(map_array != -1)

        if total_cells > 0:
            self.exploration_percentage = (explored_cells / total_cells) * 100.0

            self.get_logger().info(
                f'Exploration: {self.exploration_percentage:.2f}%',
                throttle_duration_sec=5.0
            )
        

            if not self.threshold_reached and self.exploration_percentage >= self.exploration_threshold:
                self.threshold_reached = True
                self.get_logger().info('SEUIL ATTEINT → MODE FRONTIER')

                self.find_frontier_points(map_array, msg.info)
                self.exploration_mode = 'frontier'

                stop = Twist()
                self.cmd_pub.publish(stop)

                nav_msg = Bool()
                nav_msg.data = True
                self.nav_enable_pub.publish(nav_msg)
            
            # En mode frontier, mettre à jour les frontières à chaque nouvelle carte
            if self.exploration_mode == 'frontier':
                self.find_frontier_points(map_array, msg.info)
                
                # Si la frontière actuelle n'est plus valide, passer à la suivante
                if not self.is_frontier_still_valid():
                    self.get_logger().info("Frontière invalide → passage suivant")
                    self.go_to_next_frontier()

    def update_visualization(self):
        """Mettre à jour la visualisation CV2"""
        if not self.show_visualization or self.map_data is None or self.map_info is None:
            return
        
        # Créer une image couleur de la carte
        width = self.map_info.width
        height = self.map_info.height
        grid = self.map_data.reshape((height, width))
        
        # Convertir la grille en image RGB
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Dessiner la carte
        # Gris foncé pour inconnu (-1)
        img[grid == -1] = [50, 50, 50]
        # Blanc pour libre (0-49)
        img[(grid >= 0) & (grid < 50)] = [255, 255, 255]
        # Noir pour obstacles (50-100)
        img[grid >= 50] = [0, 0, 0]
        
        # Dessiner les frontières en CYAN
        for i, point in enumerate(self.frontier_points):
            gx, gy = point['grid_x'], point['grid_y']
            
            # Point cible en VERT (plus gros)
            if i == self.current_frontier_index and self.exploration_mode == 'frontier':
                cv2.circle(img, (gx, gy), 5, (0, 255, 0), -1)  # Vert
            else:
                cv2.circle(img, (gx, gy), 2, (255, 255, 0), -1)  # Cyan
        
        # Dessiner la position du robot en ROUGE
        pose = self.get_robot_pose()
        if pose:
            x, y = pose
            resolution = self.map_info.resolution
            origin_x = self.map_info.origin.position.x
            origin_y = self.map_info.origin.position.y
            
            robot_grid_x = int((x - origin_x) / resolution)
            robot_grid_y = int((y - origin_y) / resolution)
            
            if 0 <= robot_grid_x < width and 0 <= robot_grid_y < height:
                cv2.circle(img, (robot_grid_x, robot_grid_y), 8, (0, 0, 255), -1)  # Rouge
        
        # Ajouter du texte
        text = f'Mode: {self.exploration_mode} | Frontiers: {len(self.frontier_points)} | Exploration: {self.exploration_percentage:.1f}%'
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Afficher
        cv2.imshow('Frontier Exploration', img)
        cv2.waitKey(1)
                        
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
        width = map_info.width
        height = map_info.height
        resolution = map_info.resolution
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y

        grid = map_array.reshape((height, width))
        self.frontier_points = []

        for y in range(2, height - 2):  # Marges plus grandes
            for x in range(2, width - 2):
                val = grid[y, x]

                # Frontière = cellule libre avec au moins un voisin inconnu
                if val >= 0 and val < 50:  # Cellule libre
                    neighbors = [
                        grid[y-1, x], grid[y+1, x], grid[y, x-1], grid[y, x+1]
                    ]
                    
                    # Au moins un voisin inconnu
                    has_unknown = any(n == -1 for n in neighbors)
                    # Pas d'obstacle proche
                    is_safe = self.is_frontier_safe(grid, x, y, safety_radius=3)
                    
                    if has_unknown and is_safe:
                        world_x = origin_x + (x + 0.5) * resolution
                        world_y = origin_y + (y + 0.5) * resolution
                        self.frontier_points.append({
                            'x': world_x,
                            'y': world_y,
                            'grid_x': x,
                            'grid_y': y,
                            'unknown_count': sum(1 for n in neighbors if n == -1)
                        })
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
        if self.waiting_for_goal_completion:
            return

        if len(self.frontier_points) == 0:
            self.get_logger().info('Plus de frontières - terminé')
            self.exploration_mode = 'done'
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_enable_pub.publish(nav_msg)
            return

        pose = self.get_robot_pose()
        if pose:
            self.robot_x, self.robot_y = pose
        else:
            return

        closest_idx = 0
        min_dist = float('inf')

        for i, point in enumerate(self.frontier_points):
            dist = math.sqrt(
                (point['x'] - self.robot_x)**2 +
                (point['y'] - self.robot_y)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        self.current_frontier_index = closest_idx
        target = self.frontier_points[closest_idx]

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = target['x']
        goal_msg.pose.position.y = target['y']
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)
        self.waiting_for_goal_completion = True

        self.get_logger().info(
            f'GOAL → frontière {closest_idx+1}/{len(self.frontier_points)} '
            f'dist={min_dist:.2f} m'
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
    
    def goal_reached_callback(self, msg):
        if not msg.data or not self.exploration_enabled or self.exploration_mode != 'frontier':
            return

        self.get_logger().info('✓ Frontière atteinte')

        self.waiting_for_goal_completion = False

        # Recalcul complet des frontières
        if self.map_info is not None and self.map_data is not None:
            self.get_logger().info('Recalcul des frontières...')
            self.find_frontier_points(self.map_data, self.map_info)

        if len(self.frontier_points) == 0:
            self.get_logger().info('Plus de frontières - exploration terminée')
            self.exploration_mode = 'done'
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_enable_pub.publish(nav_msg)
            return

        # Petit délai pour stabiliser TF / carte
        def delayed():
            self.go_to_next_frontier()
            timer.cancel()
            self.destroy_timer(timer)

        timer = self.create_timer(0.5, delayed)

    
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
        angle_front = int(num_points * 0.70)

        front_left = ranges[:-angle_front]
        front_right = ranges[-angle_front:]
        front_ranges = np.concatenate([front_left])
        
        if len(front_ranges) > 0:
            min_distance = np.min(front_ranges)
            self.obstacle_ahead = min_distance < self.obstacle_distance
            
    def is_frontier_still_valid(self):
        p = self.frontier_points[self.current_frontier_index]
        gx, gy = p['grid_x'], p['grid_y']
        grid = self.map_data.reshape((self.map_info.height, self.map_info.width))
        return grid[gy, gx] == -1
    
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
                # Le callback goal_reached_callback() gère l'arrivée aux frontières
                # Donc on attend juste que le callback change waiting_for_goal_completion à False
                pass
            
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
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()