#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import heapq
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
import cv2


class PathManager(Node):
    def __init__(self):
        super().__init__('path_manager')
        
        self.get_logger().info("Initialisation de PathManager...")
        
        # ========== INITIALISATION DES VARIABLES ==========
        self.map_data = None
        self.grid = None
        self.map_width = None
        self.map_height = None
        self.msg_grid = None
        self.free_cell = None
        
        # Position start et goal
        self.start_x = None
        self.start_y = None
        self.goal_x = None
        self.goal_y = None
        
        # Flags d'état
        self.nav_enabled = False
        self.path_computed = False
        self.map_received = False
        
        # ========== TF ==========
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ========== SUBSCRIBERS ==========
        qos_map = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_map
        )
        
        self.enable_sub = self.create_subscription(
            Bool,
            '/nav/enable',
            self.enable_callback,
            10
        )
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # ========== PUBLISHER ==========
        self.path_pub = self.create_publisher(Path, '/computed_path', 10)
        
        self.get_logger().info("PathManager initialisé - en attente de /map et /goal_pose")
        

    
        
    def goal_callback(self, msg: PoseStamped):
        """Recevoir un nouveau goal"""
        # Sécurité : on accepte uniquement les goals en frame map
        if msg.header.frame_id != 'map':
            self.get_logger().warn(
                f"Goal ignoré (frame = {msg.header.frame_id}, attendu = map)"
            )
            return
        
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.path_computed = False  # Autoriser un nouveau calcul

        self.get_logger().info(
            f"Nouveau goal reçu: ({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

        # Calcul immédiat SI navigation activée ET map reçue
        if self.nav_enabled and self.map_received:
            self.compute_path()
        elif not self.nav_enabled:
            self.get_logger().info("Goal enregistré - en attente d'activation de navigation")
        elif not self.map_received:
            self.get_logger().info("Goal enregistré - en attente de la map")

   
    
        
    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """Récupérer la position de départ une seule fois"""
        if self.path_computed:
            #self.get_logger().info(f"ignoring additional start position")

            return  # Ignorer si on a déjà la position
        
        self.start_x = msg.pose.pose.position.x
        self.start_y = msg.pose.pose.position.y
        self.start_received = True
        
        self.get_logger().info(f"Position initiale reçue: ({self.start_x:.2f}, {self.start_y:.2f})")
        
        # Si navigation est déjà activée, calculer le chemin
        if self.nav_enabled and self.map_received:
            self.compute_path()
            
    def dilate_grid(self, grid, radius):
        dilated = grid.copy()
        h, w = grid.shape
        obstacles = np.argwhere(grid == 1)

        for (r, c) in obstacles:
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        dilated[rr, cc] = 1
        return dilated

   

    def map_callback(self, msg: OccupancyGrid):
        """Récupérer la map"""
        if not self.nav_enabled:
            return
        
        self.get_logger().info(f"Map reçue: {msg.info.width}x{msg.info.height} @ {msg.info.resolution} m/cell")
        
        self.msg_grid = msg
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        
        grid_data = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))
        raw_grid = np.where(grid_data >= 50, 1, 0).astype(np.uint8)

        # ----- DILATATION -----
        robot_radius_m = 0.1  # 25 cm
        cell_size = msg.info.resolution
        inflation_radius = int(robot_radius_m / cell_size)

        self.grid = self.dilate_grid(raw_grid, inflation_radius)

        raw_vis = (raw_grid * 255).astype(np.uint8)
        dilated_vis = (self.grid * 255).astype(np.uint8)

        cv2.imshow("Raw map", raw_vis)
        cv2.imshow("Dilated map", dilated_vis)
        cv2.waitKey(1)
                
        free_cells = np.argwhere(self.grid == 0)
        occupied_cells = np.argwhere(self.grid == 1)
        
        self.get_logger().info(f"Dimensions: {self.map_width} x {self.map_height}")
        self.get_logger().info(f"Cellules libres: {len(free_cells)}")
        self.get_logger().info(f"Cellules occupées: {len(occupied_cells)}")
        
        self.free_cell = free_cells
        if self.free_cell.size == 0:
            self.get_logger().error("Aucune cellule libre trouvée dans la map!")
            return
        
        self.map_received = True
        self.get_logger().info("✓ Map prête pour la planification")
        
        # Si navigation est déjà activée, calculer le chemin
        if self.nav_enabled :
            self.compute_path()
        
    def enable_callback(self, msg):
        """Réagir à l'activation de la navigation"""
        self.nav_enabled = msg.data
        
        if msg.data:
            self.get_logger().info("Navigation activée - calcul du chemin")
        else:
            self.get_logger().info("Navigation désactivée")
            self.path_computed = False

        
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

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            return x, y
        except:
            return None

    
    def compute_path(self):
        """Calculer le chemin vers le goal"""
        
        # ===== VÉRIFICATIONS PRÉALABLES =====
        if not self.nav_enabled:
            self.get_logger().info("Calcul ignoré - navigation désactivée")
            return
        
        if self.path_computed:
            self.get_logger().info("Chemin déjà calculé")
            return
        
        if self.goal_x is None:
            self.get_logger().warn("Pas de goal défini")
            return
        
        if not self.map_received:
            self.get_logger().warn("Map non reçue")
            return
        
        # ===== RÉCUPÉRER POSITION ROBOT =====
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().warn("Impossible d'obtenir la pose du robot via TF")
            return

        self.start_x, self.start_y = robot_pose

        # ===== CONVERSION COORDONNÉES =====
        goal_row, goal_col = self.world_to_map(self.goal_x, self.goal_y)
        start_row, start_col = self.world_to_map(self.start_x, self.start_y)

        self.get_logger().info(f"Start (monde): ({self.start_x:.2f}, {self.start_y:.2f})")
        self.get_logger().info(f"Start (grille): ({start_row}, {start_col})")
        self.get_logger().info(f"Goal (monde): ({self.goal_x:.2f}, {self.goal_y:.2f})")
        self.get_logger().info(f"Goal (grille): ({goal_row}, {goal_col})")
        
        # ===== VÉRIFIER OBSTACLES =====
        if self.grid[start_row, start_col] == 1:
            self.get_logger().warn(f"Position de départ est un obstacle!")
            if len(self.free_cell) > 0:
                start_row, start_col = self.free_cell[0]
                self.get_logger().info(f"Nouvelle position: ({start_row}, {start_col})")
        
        if self.grid[goal_row, goal_col] == 1:
            self.get_logger().warn(f"Goal est un obstacle!")
            if len(self.free_cell) > 0:
                goal_row, goal_col = self.free_cell[0]
                self.get_logger().info(f"Nouveau goal: ({goal_row}, {goal_col})")
        
        # ===== CALCUL A* =====
        self.get_logger().info("Calcul du chemin A*...")
        path = PathManager.astar(self.grid, (start_row, start_col), (goal_row, goal_col))

        if not path:
            self.get_logger().error("AUCUN CHEMIN TROUVÉ!")
            self.path_computed = False
            return
        
        # ===== PUBLIER =====
        self.path_computed = True
        self.publish_path(path, "map")
        
        first_world = self.map_to_world(path[0][0], path[0][1])
        last_world = self.map_to_world(path[-1][0], path[-1][1])
        self.get_logger().info(f"Chemin trouvé ({len(path)} waypoints)")
        self.get_logger().info(f"  Début: {first_world}")
        self.get_logger().info(f"  Fin: {last_world}")

    def publish_path(self, path, frame_id):
        """Publier le chemin calculé"""
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
        
        
        
        #------- les méthodes de la classe -------
    @staticmethod
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    @staticmethod
    def astar(grid, start, goal, max_iterations=50000):
        # cette fonction implémente l'algorithme A* pour trouver un chemin dans une grille binaire
        #print("Goal:", goal, "valeur dans grid:", grid[goal[0], goal[1]])
        neighbors = [(0,1),(1,0),(0,-1),(-1,0)]# tous les déplacements possibles (4-connectivité)
        
        # structure du A*
        close_set = set() # 
        came_from = {} # pour reconstruire le chemin equivalent previous dans le cours
        gscore = {start:0}  # cout du mouvement depuis le départ jusqu'au noeud actuel 1 pixel = cout de 1
        fscore = {start: PathManager.heuristic(start, goal)}  # fscore = gscore + heuristique jusqu'au but
        oheap = [] # pile des noeuds à explorer ?? avec noeud = pixels

        heapq.heappush(oheap, (fscore[start], start)) # pile dans laquelle on a les noeuds explorés avec le plus petit fscore en premier
        iterations = 0 

        #------ Boucle d'exploration A* ------
        while oheap and iterations < max_iterations: # tant qu'il y a des noeuds à explorer et pas trop d'itérations on explore
            iterations += 1
            current = heapq.heappop(oheap)[1]

            #------ Vérifier si on a atteint le but ------
            if current == goal: 
                # reconstruire le chemin
                path = [] 
                while current in came_from: # tant que le noeud courant a un parent
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                print(f"A* trouvé en {iterations} itérations")
                return path[::-1]

            # ------ Sinon Explorer les voisins ------
            close_set.add(current) # ajouter le noeud courant aux explorés
            # Pour chaque voisin possible de mon pixel courant
            for i, j in neighbors: 
                neighbor = current[0]+i, current[1]+j
                tentative_g_score = gscore[current]+1
                # vérifier si le voisin est dans la grille
                if 0 <= neighbor[0] < grid.shape[0]:
                    if 0 <= neighbor[1] < grid.shape[1]:
                        # vérifier si le voisin est libre
                        if grid[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor,0): # si le voisin est déjà exploré ou pas meilleur on skip
                    continue

                if tentative_g_score < gscore.get(neighbor,0) or neighbor not in [i[1] for i in oheap]: # si le chemin vers le voisin est meilleur ou pas encore dans à explorer
                    # mettre à jour les scores et le parent
                    came_from[neighbor] = current
                    # Mettre à jour gscore et fscore
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + PathManager.heuristic(neighbor, goal)
                    # Ajouter le voisin à la pile à explorer
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        

        print(f"A* timeout après {iterations} itérations - pas de chemin trouvé")
        return []


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PathManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Arrêt par utilisateur")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()