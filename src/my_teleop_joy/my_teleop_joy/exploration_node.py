#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
import cv2
import time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

class ExplorationNode(Node):
    """
    Nœud d'exploration intelligent avec visualisation CV2.
    
    Algorithme:
    1. Exploration aléatoire jusqu'au seuil (40%)
    2. Au seuil, générer des points frontières
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
        self.exploration_mode = 'random'  # 'random', 'frontier', ou 'done'
        
        # Vitesses (mode random uniquement)
        self.forward_speed = 0.15  # m/s
        self.turn_speed = 0.4      # rad/s
        self.obstacle_distance = 0.4  # m
        
        # Suivi de l'exploration
        self.exploration_percentage = 0.0
        self.map_data = None
        self.map_info = None
        self.threshold_reached = False
        self.exploration_threshold = 40.0  # Seuil de 40%
        
        # Points frontières
        self.frontier_points = []
        self.current_frontier_index = 0
        
        # Position du robot
        self.robot_x = None
        self.robot_y = None

        # Navigation state
        self.waiting_for_goal_completion = False
        
        # Timestamp du dernier goal atteint
        self.last_goal_reached_time = None
        
        # Gestion de l'absence de map
        self.last_map_time = None
        self.forced_random_until = None
        self.saved_mode_before_forced_random = None
        
        # TF pour position du robot
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Visualisation CV2
        self.show_visualization = True
        cv2.namedWindow('Frontier Exploration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frontier Exploration', 800, 800)
        
        qos_map = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
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
            qos_map
        )
        
        self.path_sub = self.create_subscription(
            Path,
            '/computed_path',
            self.path_callback,
            10
        )
        
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
        
        # Timer de contrôle (100 Hz)
        self.timer = self.create_timer(0.01, self.control_loop)
        
        # Timer pour visualisation CV2 (2 Hz)
        self.viz_timer = self.create_timer(0.5, self.update_visualization)
        
        self.get_logger().info('Exploration Node initialisé')
        self.get_logger().info('Utilise PathManager + TrajectoryPlanner pour la navigation')
        self.get_logger().info('Visualisation CV2 activée')
        
        self.visited_frontiers = []
        self.min_frontier_distance = 0.5  # mètres

    
    def enable_callback(self, msg):
        """Activer/désactiver l'exploration"""
        self.exploration_enabled = msg.data
        
        if msg.data:
            self.get_logger().info('🚀 EXPLORATION ACTIVÉE')
            self.threshold_reached = False
            self.frontier_points = []
            self.exploration_mode = 'random'
            self.waiting_for_goal_completion = False
            self.last_map_time = time.time()
        else:
            self.get_logger().info('🛑 EXPLORATION DÉSACTIVÉE')
            # Arrêter le robot
            stop = Twist()
            self.cmd_pub.publish(stop)
            # Désactiver la navigation
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_enable_pub.publish(nav_msg)
    
    def map_callback(self, msg):
        """Traiter les mises à jour de la carte"""
        if not self.exploration_enabled:
            return
        
        # Mettre à jour le timestamp
        self.last_map_time = time.time()

        # Restaurer le mode si on était en forced-random
        if self.forced_random_until is not None:
            self.forced_random_until = None
            if self.saved_mode_before_forced_random is not None:
                self.exploration_mode = self.saved_mode_before_forced_random
                self.saved_mode_before_forced_random = None
                self.get_logger().info('Carte reçue - mode restauré')
        
        self.map_info = msg.info
        self.map_data = np.array(msg.data)

        map_array = self.map_data
        total_cells = len(map_array)
        explored_cells = np.sum(map_array != -1)

        if total_cells > 0:
            self.exploration_percentage = (explored_cells / total_cells) * 100.0

            self.get_logger().info(
                f'📊 Exploration: {self.exploration_percentage:.2f}%',
                throttle_duration_sec=5.0
            )
        
            # Atteindre le seuil → passer en mode frontier
            if not self.threshold_reached and self.exploration_percentage >= self.exploration_threshold:
                self.threshold_reached = True
                self.get_logger().info('🎯 SEUIL ATTEINT → MODE FRONTIER')

                self.find_frontier_points(map_array, msg.info)
                
                if len(self.frontier_points) == 0:
                    self.get_logger().warn('Aucune frontière trouvée - rester en mode random')
                    return
                
                self.exploration_mode = 'frontier'

                # Arrêter le mouvement aléatoire
                stop = Twist()
                self.cmd_pub.publish(stop)

                # Activer la navigation
                nav_msg = Bool()
                nav_msg.data = True
                self.nav_enable_pub.publish(nav_msg)
                
                # Envoyer le premier but frontier
                self.go_to_next_frontier()
            
            # En mode frontier, mettre à jour les frontières régulièrement
            if self.exploration_mode == 'frontier':
                old_count = len(self.frontier_points)
                self.find_frontier_points(map_array, msg.info)
                
                if len(self.frontier_points) != old_count:
                    self.get_logger().info(
                        f'🔄 Frontières mises à jour: {len(self.frontier_points)}',
                        throttle_duration_sec=2.0
                    )
                
                # Si la frontière actuelle n'est plus valide, passer à la suivante
                if self.waiting_for_goal_completion and not self.is_frontier_still_valid():
                    self.get_logger().warn("⚠️ Frontière invalide → passage à la suivante")
                    
                    # Arrêter le robot
                    stop = Twist()
                    self.cmd_pub.publish(stop)
                    
                    # Désactiver temporairement la navigation
                    nav_msg = Bool()
                    nav_msg.data = False
                    self.nav_enable_pub.publish(nav_msg)
                    
                    # Passer à la frontière suivante
                    self.waiting_for_goal_completion = False
                    
                    # Petit délai avant le prochain goal
                    def delayed_next():
                        nav_msg = Bool()
                        nav_msg.data = True
                        self.nav_enable_pub.publish(nav_msg)
                        self.go_to_next_frontier()
                        timer.cancel()
                        self.destroy_timer(timer)
                    
                    timer = self.create_timer(0.5, delayed_next)

    def update_visualization(self):
        """Mettre à jour la visualisation CV2"""
        if not self.show_visualization or self.map_data is None or self.map_info is None:
            return
        
        width = self.map_info.width
        height = self.map_info.height
        grid = self.map_data.reshape((height, width))
        
        # Convertir en image RGB
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gris foncé pour inconnu (-1)
        img[grid == -1] = [50, 50, 50]
        # Blanc pour libre (0-49)
        img[(grid >= 0) & (grid < 50)] = [255, 255, 255]
        # Noir pour obstacles (50-100)
        img[grid >= 50] = [0, 0, 0]
        
        # Dessiner les frontières
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
        
        # Texte d'information
        text = f'Mode: {self.exploration_mode} | Frontiers: {len(self.frontier_points)} | Exploration: {self.exploration_percentage:.1f}%'
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Afficher
        cv2.imshow('Frontier Exploration', img)
        cv2.waitKey(1)
                        
    def is_frontier_safe(self, grid, x, y, safety_radius=3):
        """Vérifier qu'il n'y a pas d'obstacle proche de la frontière"""
        height, width = grid.shape
        
        for dy in range(-safety_radius, safety_radius + 1):
            for dx in range(-safety_radius, safety_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny, nx] >= 50:  # Obstacle
                        return False
        return True
    
    def find_frontier_points(self, map_array, map_info):
        """
        Trouver les points frontières (cellules libres adjacentes à l'inconnu)
        """
        width = map_info.width
        height = map_info.height
        resolution = map_info.resolution
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y

        grid = map_array.reshape((height, width))
        self.frontier_points = []

        for y in range(3, height - 3):
            for x in range(3, width - 3):
                val = grid[y, x]

                # Frontière = cellule LIBRE (blanche) avec au moins un voisin INCONNU (gris)
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
        
        self.get_logger().info(
            f'🔍 {len(self.frontier_points)} frontières détectées',
            throttle_duration_sec=3.0
        )
    
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
        except Exception:
            return None
    
    def go_to_next_frontier(self):
        """Envoyer le robot vers la meilleure frontière valide"""

        if self.waiting_for_goal_completion:
            return

        if len(self.frontier_points) == 0:
            self.get_logger().info('✅ Plus de frontières - exploration terminée')
            self.exploration_mode = 'done'
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_enable_pub.publish(nav_msg)
            return

        pose = self.get_robot_pose()
        if not pose:
            self.get_logger().warn('Position robot indisponible')
            return

        self.robot_x, self.robot_y = pose

        best_idx = None
        best_dist = float('inf')

        for i, point in enumerate(self.frontier_points):
            # Distance monde
            dist = math.hypot(
                point['x'] - self.robot_x,
                point['y'] - self.robot_y
            )

            # 1. Trop proche → ignorer
            if dist < self.min_frontier_distance:
                continue

            # 2. Déjà visitée → ignorer
            grid_key = (point['grid_x'], point['grid_y'])
            if grid_key in self.visited_frontiers:
                continue

            # 3. Garder la meilleure
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Aucune frontière valide restante
        if best_idx is None:
            self.get_logger().info('✅ Plus de frontières valides - exploration terminée')
            self.exploration_mode = 'done'
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_enable_pub.publish(nav_msg)
            return

        # Sélection finale
        self.current_frontier_index = best_idx
        target = self.frontier_points[best_idx]

        # Publier le goal
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = target['x']
        goal_msg.pose.position.y = target['y']
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)
        self.waiting_for_goal_completion = True

        self.get_logger().info(
            f'🎯 GOAL → frontière {best_idx+1}/{len(self.frontier_points)} '
            f'à {best_dist:.2f}m'
        )

    
    def path_callback(self, msg):
        """Recevoir le chemin calculé (info seulement)"""
        if not self.exploration_enabled or self.exploration_mode != 'frontier':
            return
        
        if len(msg.poses) > 0:
            self.get_logger().info(
                f'✓ Chemin reçu: {len(msg.poses)} waypoints',
                throttle_duration_sec=2.0
            )
    
    def goal_reached_callback(self, msg):
        """Callback quand le robot atteint le goal"""
        if not msg.data or not self.exploration_enabled or self.exploration_mode != 'frontier':
            return

        self.get_logger().info('✅ Frontière atteinte')
        self.waiting_for_goal_completion = False

        # Recalcul des frontières
        if self.map_info is not None and self.map_data is not None:
            self.find_frontier_points(self.map_data, self.map_info)

        if len(self.frontier_points) == 0:
            self.get_logger().info('✅ Exploration terminée - plus de frontières')
            self.exploration_mode = 'done'
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_enable_pub.publish(nav_msg)
            return

        # Délai avant le prochain goal
        def delayed():
            self.go_to_next_frontier()
            timer.cancel()
            self.destroy_timer(timer)

        timer = self.create_timer(1.0, delayed)
    
    def scan_callback(self, msg):
        """Détecter les obstacles devant (mode random uniquement)"""
        if not self.exploration_enabled or self.exploration_mode != 'random':
            return
        
        ranges = np.array(msg.ranges)
        ranges = ranges[~np.isnan(ranges)]
        ranges = ranges[ranges > 0]
        
        if len(ranges) == 0:
            return
        
        # Vérifier le secteur avant (±30°)
        num_points = len(ranges)
        angle_front = int(num_points * 0.15)  # 15% de chaque côté = 30°
        
        front_ranges = np.concatenate([
            ranges[:angle_front],
            ranges[-angle_front:]
        ])
        
        if len(front_ranges) > 0:
            min_distance = np.min(front_ranges)
            self.obstacle_ahead = min_distance < self.obstacle_distance
    
    def is_frontier_still_valid(self):
        """Vérifier si la frontière actuelle est toujours valide"""
        if not self.frontier_points:
            return False
        if self.current_frontier_index < 0 or self.current_frontier_index >= len(self.frontier_points):
            return False

        p = self.frontier_points[self.current_frontier_index]
        gx, gy = p['grid_x'], p['grid_y']

        if self.map_data is None or self.map_info is None:
            return False

        height = self.map_info.height
        width = self.map_info.width
        
        if gx < 0 or gy < 0 or gx >= width or gy >= height:
            return False

        grid = self.map_data.reshape((height, width))

        # La cellule doit rester libre
        val = grid[gy, gx]
        if not (val >= 0 and val < 50):
            return False

        # Doit avoir au moins un voisin inconnu
        neighbors = [
            grid[gy-1, gx], grid[gy+1, gx], grid[gy, gx-1], grid[gy, gx+1]
        ]
        has_unknown = any(n == -1 for n in neighbors)
        if not has_unknown:
            return False

        # Vérifier la sécurité
        return self.is_frontier_safe(grid, gx, gy, safety_radius=3)
    
    def control_loop(self):
        """Boucle de contrôle principale"""
        if not self.exploration_enabled:
            return
        
        # Vérifier l'absence de map
        now = time.time()
        
        # Si période forcée 'random' terminée
        if self.forced_random_until is not None:
            if now >= self.forced_random_until:
                self.forced_random_until = None
                if self.saved_mode_before_forced_random is not None:
                    self.exploration_mode = self.saved_mode_before_forced_random
                    self.saved_mode_before_forced_random = None
                    self.get_logger().info('Période random forcée terminée')
        else:
            # Si pas de map depuis >5s, forcer random pendant 3s
            if self.last_map_time is None or (now - self.last_map_time) > 5.0:
                if self.exploration_mode != 'random':
                    self.saved_mode_before_forced_random = self.exploration_mode
                self.exploration_mode = 'random'
                self.forced_random_until = now + 3.0
                self.get_logger().warn('⚠️ Pas de map depuis >5s — mode random forcé')
        
        twist = Twist()
        
        # Mode exploration aléatoire
        if self.exploration_mode == 'random':
            if self.obstacle_ahead:
                twist.angular.z = self.turn_speed
            else:
                twist.linear.x = self.forward_speed
            
            self.cmd_pub.publish(twist)
        
        # Mode navigation vers frontières
        elif self.exploration_mode == 'frontier':
            # TrajectoryPlanner gère le mouvement
            pass
        
        # Mode terminé
        elif self.exploration_mode == 'done':
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