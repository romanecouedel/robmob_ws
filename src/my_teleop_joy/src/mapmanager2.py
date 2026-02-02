#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import heapq

from tf2_ros import Buffer, TransformListener


class PathManager(Node):
    def __init__(self, goal_world):
        super().__init__('path_manager')
        
        # Configuration TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Configurer le QoS pour la subscription de la map
        qos_map = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_map)
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )   
        
        # ✨ Publishers pour les deux chemins
        self.path_pub = self.create_publisher(Path, '/computed_path', 10)
        self.return_path_pub = self.create_publisher(Path, '/return_path', 10)
        
        # ✨ Publisher pour visualiser la map traitée
        self.processed_map_pub = self.create_publisher(OccupancyGrid, '/map_processed', 10)
        
        # Initialisation de variables
        self.map_received = False
        self.grid = None
        self.map_width = None
        self.map_height = None
        self.msg_grid = None
        self.free_cell = None
        
        # ✨ PARAMÈTRES DE TRAITEMENT DES ZONES NON EXPLORÉES
        self.unknown_threshold = 50  # Valeur seuil pour déterminer "inconnu"
                                     # 0-50 = explorée (libre)
                                     # 50-100 = non explorée ou occupée
        
        self.treat_unknown_as_obstacle = True  # Traiter zones non explorées comme obstacles
        
        self.inflation_radius = 2    # Rayon de dilatation autour des zones inconnues
        
        # Goal
        self.goal_x = goal_world[0]
        self.goal_y = goal_world[1]
        self.path_computed = False
        
        self.get_logger().info("🗺️ PathManager initialized with unknown zone handling")
        self.get_logger().info(f"Unknown threshold: {self.unknown_threshold}")
        self.get_logger().info(f"Treat unknown as obstacle: {self.treat_unknown_as_obstacle}")
        self.get_logger().info(f"Inflation radius: {self.inflation_radius}")
        
        # Attendre seulement la map
        while not self.map_received:
            self.get_logger().info("En attente de la map...")
            rclpy.spin_once(self, timeout_sec=1.0)

        self.get_logger().info("✓ Map reçue et traitée - prêt à planifier")
        
    def get_robot_pose_from_tf(self):
        """Récupérer la position du robot via TF map->base_footprint"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                rclpy.time.Time()
            )
            
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            return (x, y)
            
        except Exception as e:
            self.get_logger().warn(f"⚠️ TF indisponible: {e}")
            return None

    def goal_callback(self, msg: PoseStamped):
        """Recevoir un nouveau goal"""
        if msg.header.frame_id != 'map':
            self.get_logger().warn(
                f"Goal ignoré (frame = {msg.header.frame_id}, attendu = map)"
            )
            return
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.path_computed = False

        self.get_logger().info(
            f"Nouveau goal reçu: ({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

        if self.map_received:
            self.compute_path()

    def map_callback(self, msg: OccupancyGrid):
        """Recevoir et traiter la map"""
        if self.map_received:
            return
        
        self.get_logger().info(f"Map reçue: {msg.info.width}x{msg.info.height} @ {msg.info.resolution} m/cell")
        
        self.msg_grid = msg
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        
        # ✨ Traiter la map : convertir les zones non explorées en obstacles
        grid_data = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))
        self.grid = self.process_unknown_zones(grid_data)
        
        # ✨ Publier la map traitée pour visualisation
        self.publish_processed_map(self.grid)
        
        free_cells = np.argwhere(self.grid == 0)
        occupied_cells = np.argwhere(self.grid == 1)
        
        self.get_logger().info(f"Dimensions: {self.map_width} x {self.map_height}")
        self.get_logger().info(f"Cellules libres: {len(free_cells)}")
        self.get_logger().info(f"Cellules occupées/inconnues: {len(occupied_cells)}")
        
        self.free_cell = free_cells
        if self.free_cell.size == 0:
            self.get_logger().error("Aucune cellule libre trouvée dans la map!")
            return
        
        self.map_received = True
        self.get_logger().info("✓ Map traitée et prête pour la planification")

    def process_unknown_zones(self, grid_data):
        """
        ✨ Traiter les zones non explorées comme obstacles
        
        Valeurs dans OccupancyGrid :
        -1 = inconnu (gris)
        0-49 = libre (blanc)
        50-100 = occupé (noir)
        
        Stratégie :
        1. Seuillage : >= unknown_threshold → obstacle
        2. Optionnel : Dilatation pour sécurité
        """
        
        self.get_logger().info("🔄 Traitement des zones non explorées...")
        
        # ✨ Étape 1 : Seuillage
        # Traiter les zones grises ET noires comme obstacles
        if self.treat_unknown_as_obstacle:
            # >= unknown_threshold = obstacle (comprend -1, gris, noir)
            binary_grid = (grid_data >= self.unknown_threshold).astype(np.uint8)
            
            obstacles_detected = np.sum(binary_grid)
            self.get_logger().info(
                f"✓ Seuillage: {self.unknown_threshold} → "
                f"{obstacles_detected} cellules traitées comme obstacles"
            )
        else:
            # Seuil standard (>= 50)
            binary_grid = (grid_data >= 50).astype(np.uint8)
        
        # ✨ Étape 2 : Dilatation optionnelle (zone de sécurité)
        if self.inflation_radius > 0:
            binary_grid = self.dilate_obstacles(binary_grid)
            self.get_logger().info(
                f"✓ Dilatation appliquée (rayon={self.inflation_radius})"
            )
        
        return binary_grid

    def dilate_obstacles(self, grid):
        """
        ✨ Dilater les obstacles pour créer une zone de sécurité
        
        Évite que le robot passe trop proche des zones inconnues
        """
        from scipy.ndimage import binary_dilation
        
        dilated = grid.copy()
        
        for _ in range(self.inflation_radius):
            # Dilater dans toutes les directions
            dilated = binary_dilation(dilated).astype(np.uint8)
        
        return dilated

    def publish_processed_map(self, processed_grid):
        """
        ✨ Publier la map traitée pour la visualiser dans RViz
        """
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = self.msg_grid.header.frame_id
        map_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Copier les métadonnées
        map_msg.info = self.msg_grid.info
        
        # Convertir en occupancy grid
        map_data = []
        for val in processed_grid.flatten():
            if val == 1:
                map_data.append(100)  # Occupé (noir)
            else:
                map_data.append(0)    # Libre (blanc)
        
        map_msg.data = map_data
        self.processed_map_pub.publish(map_msg)

    def map_to_world(self, row, col):
        """Convertir indices grille → coordonnées monde"""
        origin_x = self.msg_grid.info.origin.position.x
        origin_y = self.msg_grid.info.origin.position.y
        res = self.msg_grid.info.resolution
        
        x = origin_x + col * res
        y = origin_y + row * res
        
        return (x, y)

    def world_to_map(self, x, y):
        """Convertir coordonnées monde → indices grille"""
        origin_x = self.msg_grid.info.origin.position.x
        origin_y = self.msg_grid.info.origin.position.y
        res = self.msg_grid.info.resolution
        
        col = int((x - origin_x) / res)
        row = int((y - origin_y) / res)
        
        col = max(0, min(col, self.map_width - 1))
        row = max(0, min(row, self.map_height - 1))
        
        return (row, col)

    def compute_path(self):
        """Calculer le chemin vers le goal"""
        
        if self.path_computed:
            return
        
        robot_pose = self.get_robot_pose_from_tf()
        
        if robot_pose is None:
            self.get_logger().error("❌ Impossible de récupérer la position du robot via TF!")
            return
        
        start_x, start_y = robot_pose
        
        goal_row, goal_col = self.world_to_map(self.goal_x, self.goal_y)
        start_row, start_col = self.world_to_map(start_x, start_y)

        self.get_logger().info(f"Position départ (monde): ({start_x:.2f}, {start_y:.2f})")
        self.get_logger().info(f"Position départ (grille): ({start_row}, {start_col})")
        self.get_logger().info(f"Goal (grille): ({goal_row}, {goal_col})")
        self.get_logger().info(f"Calcul du chemin...")
        
        # Vérifier que start et goal sont libres
        if self.grid[start_row, start_col] == 1:
            self.get_logger().warn(f"Position de départ est un obstacle!")
            if len(self.free_cell) > 0:
                start_row, start_col = self.free_cell[0]
                self.get_logger().info(f"Nouvelle position de départ: ({start_row}, {start_col})")
        
        if self.grid[goal_row, goal_col] == 1:
            self.get_logger().warn(f"Goal est un obstacle!")
            if len(self.free_cell) > 0:
                goal_row, goal_col = self.free_cell[0]
                self.get_logger().info(f"Nouveau goal: ({goal_row}, {goal_col})")
        
        # Calculer le chemin
        path = PathManager.astar(self.grid, (start_row, start_col), (goal_row, goal_col))

        if not path:
            self.get_logger().error("AUCUN CHEMIN TROUVÉ!")
            self.path_computed = False
            return
        
        self.path_computed = True
        
        # Publier les chemins
        self.publish_path(path, "map")
        self.publish_return_path(path, "map")
        
        first_world = self.map_to_world(path[0][0], path[0][1])
        last_world = self.map_to_world(path[-1][0], path[-1][1])
        self.get_logger().info(f"✓ Chemin trouvé ({len(path)} waypoints)")
        self.get_logger().info(f"  Début (monde): {first_world}")
        self.get_logger().info(f"  Fin (monde): {last_world}")

    def is_goal_mode(self, current_x, current_y):
        """Déterminer si on est en mode goal ou retour"""
        dist_to_goal = ((current_x - self.goal_x)**2 + (current_y - self.goal_y)**2)**0.5
        
        if dist_to_goal < 0.5:
            return False  # Mode retour
        else:
            return True   # Mode goal

    def publish_path(self, path, frame_id):
        """Publier le chemin calculé (aller)"""
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for row, col in path:
            x, y = self.map_to_world(row, col)
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

    def publish_return_path(self, forward_path, frame_id):
        """Publier le chemin de retour"""
        return_path = forward_path[::-1]
        
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for row, col in return_path:
            x, y = self.map_to_world(row, col)
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.return_path_pub.publish(path_msg)
        
    @staticmethod
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    @staticmethod
    def astar(grid, start, goal, max_iterations=50000):
        """Implémente l'algorithme A*"""
        neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
        
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start: PathManager.heuristic(start, goal)}
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))
        iterations = 0

        while oheap and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(oheap)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                print(f"A* trouvé en {iterations} itérations")
                return path[::-1]

            close_set.add(current)
            
            for i, j in neighbors:
                neighbor = current[0]+i, current[1]+j
                tentative_g_score = gscore[current]+1
                
                if 0 <= neighbor[0] < grid.shape[0]:
                    if 0 <= neighbor[1] < grid.shape[1]:
                        if grid[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor,0):
                    continue

                if tentative_g_score < gscore.get(neighbor,0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + PathManager.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        print(f"A* timeout après {iterations} itérations - pas de chemin trouvé")
        return []


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PathManager(
            goal_world=(2.0, 2.0)
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Arrêt par utilisateur")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()