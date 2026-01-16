#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import heapq

    #------- les méthodes de la classe -------
def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal, max_iterations=50000):
        # cette fonction implémente l'algorithme A* pour trouver un chemin dans une grille binaire
        #print("Goal:", goal, "valeur dans grid:", grid[goal[0], goal[1]])
        neighbors = [(0,1),(1,0),(0,-1),(-1,0)]# tous les déplacements possibles (4-connectivité)
        
        # structure du A*
        close_set = set() # 
        came_from = {} # pour reconstruire le chemin equivalent previous dans le cours
        gscore = {start:0}  # cout du mouvement depuis le départ jusqu'au noeud actuel 1 pixel = cout de 1
        fscore = {start: self.heuristic(start, goal)}  # fscore = gscore + heuristique jusqu'au but
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
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    # Ajouter le voisin à la pile à explorer
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        

        print(f"A* timeout après {iterations} itérations - pas de chemin trouvé")
        return []


class PathManager(Node):
    def __init__(self, goal_world, start_world):
        super().__init__('path_manager')
#SUBSCRBER
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
# PUBLISHER
# Configurer le publisher pour le chemin calculé
        self.path_pub = self.create_publisher(Path, '/computed_path', 10)

# INITIALISATION de VARIABLES
        self.map_data = None
        self.map_received = False
        
        self.grid = None
        self.map_width = None
        self.map_height = None
        self.msg_grid = None
        self.free_cell = None
        
        # Goal défini en paramètres
        self.goal_x = goal_world[0]
        self.goal_y = goal_world[1]
        self.path_computed = False
        
        # Start défini en paramètres
        self.start_x = start_world[0]
        self.start_y = start_world[1]
        
        self.get_logger().info("MapManager initialized - waiting for /map")
        self.get_logger().info(f"Goal: ({self.goal_x}, {self.goal_y})")

    def map_callback(self, msg: OccupancyGrid):
        """Récupérer la map"""
        if self.map_received:
            return
        
        self.get_logger().info(f"Map reçue: {msg.info.width}x{msg.info.height} @ {msg.info.resolution} m/cell")
        
        self.msg_grid = msg
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        
        grid_data = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))
        self.grid = np.where(grid_data >= 50, 1, 0).astype(np.uint8)
        
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
        
        # Calculer le chemin immédiatement après réception de la map
        self.compute_path()

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
        
        # Position de départ fixe (0, 0)
        
        goal_row, goal_col = self.world_to_map(self.goal_x, self.goal_y)
        start_row, start_col = self.world_to_map(self.start_x, self.start_y)

        self.get_logger().info(f"Position départ (monde): ({self.start_x:.2f}, {self.start_y:.2f})")
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
        path = astar(self.grid, (start_row, start_col), (goal_row, goal_col))

        if not path:
            self.get_logger().error("AUCUN CHEMIN TROUVÉ! Robot arrêté.")
            self.path_computed = False
            return
        
        self.path_computed = True
        
        # Publier le chemin
        self.publish_path(path, "map")
        
        first_world = self.map_to_world(path[0][0], path[0][1])
        last_world = self.map_to_world(path[-1][0], path[-1][1])
        self.get_logger().info(f"✓ Chemin trouvé ({len(path)} waypoints)")
        self.get_logger().info(f"  Début (monde): {first_world}")
        self.get_logger().info(f"  Fin (monde): {last_world}")

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
        
    

def main(args=None):
    rclpy.init(args=args)
    try:
        node = PathManager(
            goal_world=(1.0, 2.0),
            start_world=(0.3, 0.3)
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Arrêt par utilisateur")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()