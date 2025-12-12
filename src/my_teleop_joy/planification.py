import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import heapq
from ament_index_python.packages import get_package_share_directory
import os

# function
# retourne la distance de Manhattan entre deux points a et b [O]=y, [1]=x


class GenerateTraj(Node):
    
    """
    GenerateTraj (Node)
 ├── Subscriber: /map  (OccupancyGrid)
 ├── Service /plan_path (PoseStamped start, PoseStamped goal → Path)
 ├── Méthode: world_to_map()
 ├── Méthode: build_grid_from_map()
 ├── Méthode: astar()
 └── Publie: /planned_path (nav_msgs/Path)
    """
    def __init__(self, map_path, goal_world):
        super().__init__('generate_traj_node')

        # Charger la carte
        self.map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map is None:
            raise FileNotFoundError(f"Carte non trouvée à l'emplacement: {map_path}")
        self.get_logger().info(f"Carte chargée depuis: {map_path}")

        # Convertir la carte en grille binaire (0=obstacle, 1=libre)
        self.map_height, self.map_width = map_img.shape
        self.grid = np.where(map_img > 127, 0, 1)  # 1=obstacle, 0=libre
        free_cells = np.argwhere(self.grid == 0)
        occupied_cells = np.argwhere(self.grid == 1)
        print(f"Nombre de cellules libres: {len(free_cells)}")
        print(f"Nombre de cellules occupées: {len(occupied_cells)}")
       

        # Convertir les coordonnées du but du monde aux indices de la grille
        # Map resolution
        self.map_resolution = map_resolution

        # OccupancyGrid message
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"
        self.map_msg.info.resolution = map_resolution
        self.map_msg.info.width = self.map_width
        self.map_msg.info.height = self.map_height
        self.map_msg.info.origin.position.x = 0.0
        self.map_msg.info.origin.position.y = 0.0
        self.map_msg.info.origin.orientation.w = 1.0
        self.map_msg.data = np.where(self.grid == 1, 100, 0).flatten().tolist()

        #choix du goal
        goal_map = self.world_to_map(*goal_world)
        
        if self.grid[goal_map[0], goal_map[1]] == 1:
            # Si objectif est sur obstacle, choisir la cellule libre la plus proche
            distances = np.sum((free_cells - goal_map)**2, axis=1)
            closest_free = free_cells[np.argmin(distances)]
            self.goal_map = (closest_free[0], closest_free[1])
        else:
            self.goal_map = goal_map
        print(f"Goal map choisi: {self.goal_map}")
        
        # Path
        self.path = []


        # Définir le point de départ (supposé être au centre bas de la carte)
        start_x_idx = int((-self.origin[0]) / self.resolution)
        start_y_idx = int(( -self.origin[1]) / self.resolution)
        start_idx = (start_y_idx, start_x_idx)  # (row, col)

        # Générer la trajectoire avec A*
        path = astar(self.grid, start_idx, goal_idx)
        if not path:
            raise ValueError("Aucun chemin trouvé vers le but spécifié.")

        # Afficher la trajectoire sur la carte pour vérification
        for point in path:
            cv2.circle(self.map, (point[1], point[0]), 1, (127), -1)  # dessiner en gris

        cv2.imshow("Trajectoire générée", self.map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
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
        fscore = {start:heuristic(start, goal)}  # fscore = gscore + heuristique jusqu'au but
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
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    # Ajouter le voisin à la pile à explorer
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        print(f"A* timeout après {iterations} itérations - pas de chemin trouvé")
        return []

    def world_to_map(self, x, y):
        """Convertir coordonnées monde (x, y) → indices grille (row, col)"""
        origin_x = self.map_msg.info.origin.position.x
        origin_y = self.map_msg.info.origin.position.y
        res = self.map_msg.info.resolution
        
        col = int((x - origin_x) / res)
        row = int((y - origin_y) / res)
        
        # Clamper aux limites
        col = max(0, min(col, self.map_width - 1))
        row = max(0, min(row, self.map_height - 1))
        
        return (row, col)

    def map_to_world(self, row, col):
        """Convertir indices grille (row, col) → coordonnées monde (x, y)"""
        origin_x = self.map_msg.info.origin.position.x
        origin_y = self.map_msg.info.origin.position.y
        res = self.map_msg.info.resolution
        
        x = origin_x + col * res
        y = origin_y + row * res
        
        return (x, y)
    
    def build_grid_from_map(self, map_msg):
        """Construire une grille binaire à partir d'un message OccupancyGrid"""
        width = map_msg.info.width
        height = map_msg.info.height
        data = np.array(map_msg.data).reshape((height, width))
        
        # Grille binaire: 0=libre, 1=obstacle
        grid = np.where(data > 50, 1, 0)
        return grid

# -----------------------------
# Main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)

    map_path = os.path.expanduser("~/robmob_ws/src/map/map_inflated.pgm")
    print(f"Utilisation de la map: {map_path}")

    traj = None

    try:
        traj = GenerateTraj(
            map_path,
            goal_world=(0.1, 2.1)
        )
        rclpy.spin(navigtrajator)
    except KeyboardInterrupt:
        print("Arrêt par utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        traj.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()