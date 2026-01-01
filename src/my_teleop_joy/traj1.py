import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import traceback
from tf2_ros import TransformListener, Buffer
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import math
import heapq

# -----------------------------
# Helper A functions
# -----------------------------
def heuristic(a, b):
    """Manhattan distance"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal, max_iterations=50000):
    """
    A* pathfinding algorithm.
    grid: 2D numpy array where 0=free, 1=obstacle
    start, goal: tuples (row, col)
    """
    print(f"A* Start: {start}, Goal: {goal}")
    
    # Vérifier que goal est accessible
    if grid[goal[0], goal[1]] != 0:
        print(f"ERROR: Goal {goal} is not free (value={grid[goal[0], goal[1]]})")
        return []
    
    # Vérifier que start est accessible
    if grid[start[0], start[1]] != 0:
        print(f"ERROR: Start {start} is not free (value={grid[start[0], start[1]]})")
        return []
    
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
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
            print(f"A* found path in {iterations} iterations, length={len(path)}")
            return path[::-1]
        
        if current in close_set:
            continue
        
        close_set.add(current)
        
        for di, dj in neighbors:
            neighbor = (current[0] + di, current[1] + dj)
            
            # Vérifier les limites
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            
            # Vérifier si obstacle
            if grid[neighbor[0], neighbor[1]] != 0:  # 0=free, anything else=obstacle
                continue
            
            # Vérifier si déjà visité
            if neighbor in close_set:
                continue
            
            tentative_g_score = gscore[current] + 1
            
            if neighbor not in gscore or tentative_g_score < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    print(f"A* timeout after {iterations} iterations - no path found")
    return []
class Trajectoire(Node):
    def __init__(self, goal_x, goal_y):
        super().__init__('trajectory_planner')

        # Publisher: velocity commands
        self.cmd_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

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

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        # TF2 for getting robot pose in map frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_data = None
        self.pose = None
        self.scan = None
        self.iteration_count = 0
        self.map_received = False
        
        self.grid = None
        self.map_width = None
        self.map_height = None
        self.msg_grid = None
        self.free_cell = None
        self.path = []
        self.path_computed = False
        
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        
        self.get_logger().info(f"Trajectory planner initialized. Waiting for map on /map topic...")
        self.get_logger().info(f"Goal: ({self.goal_x}, {self.goal_y})")

        # Timer for control loop
        self.create_timer(0.5, self.cmd)


    def map_callback(self, msg: OccupancyGrid):
        """Récupérer la map du map_server via le topic /map"""
        if self.map_received:
            return
        
        self.get_logger().info(f"📍 Map reçue: {msg.info.width}x{msg.info.height} @ {msg.info.resolution} m/cell")
        
        self.msg_grid = msg
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        
        # Convertir OccupancyGrid en numpy array
        # Les valeurs 100 = obstacle, 0 = libre, -1 = inconnu
        grid_data = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))
        
        # Normaliser: 1 = obstacle, 0 = libre
        self.grid = np.where(grid_data >= 50, 1, 0).astype(np.uint8)
        
        free_cells = np.argwhere(self.grid == 0)
        occupied_cells = np.argwhere(self.grid == 1)
        
        self.get_logger().info(f"Dimensions: {self.map_width} x {self.map_height}")
        self.get_logger().info(f"Cellules libres: {len(free_cells)}")
        self.get_logger().info(f"Cellules occupées: {len(occupied_cells)}")
        
        self.free_cell = free_cells
        if self.free_cell.size == 0:
            self.get_logger().error("❌ Aucune cellule libre trouvée dans la map!")
            return
        
        self.map_received = True
        self.get_logger().info(f"✓ Map prête pour la planification")


    def visualize_path(self, grid, path, start, goal):
        """Créer une image avec obstacles, chemin, start et goal"""
        print(f"\n=== DEBUG VISUALIZE ===")
        print(f"Grid shape: {grid.shape}")
        print(f"Path length: {len(path) if path else 0}")
        print(f"Start: {start}, Goal: {goal}")
        print(f"Path empty? {len(path) == 0}")
        
        # Créer une image en couleur
        viz = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

        # Obstacles en noir, free en blanc
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 1:
                    viz[i, j] = [0, 0, 0]  # Noir (obstacle)
                else:
                    viz[i, j] = [255, 255, 255]  # Blanc (libre)

        # Dessiner la trajectoire en bleu (ligne continue)
        if len(path) > 1:
            for i in range(len(path) - 1):
                pt1 = (path[i][1], path[i][0])
                pt2 = (path[i+1][1], path[i+1][0])
                cv2.line(viz, pt1, pt2, (255, 0, 0), 2)

        # Marquer les waypoints tous les 5 points en vert
        for i, (r, c) in enumerate(path):
            if i % 5 == 0:
                cv2.circle(viz, (c, r), 2, (0, 255, 0), -1)

        # Start en bleu clair (gros cercle)
        cv2.circle(viz, (start[1], start[0]), 8, (255, 128, 0), -1)
        cv2.circle(viz, (start[1], start[0]), 8, (0, 0, 0), 2)

        # Goal en rouge (gros cercle)
        cv2.circle(viz, (goal[1], goal[0]), 8, (0, 0, 255), -1)
        cv2.circle(viz, (goal[1], goal[0]), 8, (0, 0, 0), 2)

        # Ajouter du texte
        cv2.putText(viz, "START", (start[1]-20, start[0]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(viz, "GOAL", (goal[1]-15, goal[0]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(viz, f"Path: {len(path)} waypoints", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.namedWindow('Path Visualization', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Path Visualization', 800, 750)
        cv2.imshow('Path Visualization', viz)
        cv2.waitKey(1)

        self.get_logger().info(f"Path visualization displayed")
        self.get_logger().info(f"  Bleu = trajectoire")
        self.get_logger().info(f"  Vert = waypoints")
        self.get_logger().info(f"  Bleu clair = START")
        self.get_logger().info(f"  Rouge = GOAL")
        print("=== FIN VISUALIZE ===")
        
    def map_to_world(self, row, col):
        """Convertir indices grille (row, col) → coordonnées monde (x, y)"""
        origin_x = self.msg_grid.info.origin.position.x
        origin_y = self.msg_grid.info.origin.position.y
        res = self.msg_grid.info.resolution
        
        x = origin_x + col * res
        y = origin_y + row * res
        
        return (x, y)

    def world_to_map(self, x, y):
        """Convertir coordonnées monde (x, y) → indices grille (row, col)"""
        origin_x = self.msg_grid.info.origin.position.x
        origin_y = self.msg_grid.info.origin.position.y
        res = self.msg_grid.info.resolution
        
        col = int((x - origin_x) / res)
        row = int((y - origin_y) / res)
        
        # Clamper aux limites
        col = max(0, min(col, self.map_width - 1))
        row = max(0, min(row, self.map_height - 1))
        
        return (row, col)


    def get_yaw(self, q):
        """Extraire yaw d'un quaternion"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def odom_callback(self, msg: Odometry):
        self.pose = msg.pose.pose

    def scan_callback(self, msg):
        self.scan = msg

    def _normalize_angle(self, angle):
        """Normaliser un angle entre -π et +π"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def cmd(self):
        """Contrôle du robot pour suivre le chemin planifié"""
        
        # Attendre la réception de la map
        if not self.map_received:
            return
        
        # Attendre la pose du robot
        if self.pose is None:
            return
        
        # Calculer le chemin une seule fois (au premier appel)
        if not self.path_computed:
            rx = self.pose.position.x
            ry = self.pose.position.y
            current_yaw = self.get_yaw(self.pose.orientation)
            
            # Calculer les positions en grille
            goal_row, goal_col = self.world_to_map(self.goal_x, self.goal_y)
            start_row, start_col = self.world_to_map(rx, ry)

            self.get_logger().info(f"Position robot (monde): ({rx:.2f}, {ry:.2f})")
            self.get_logger().info(f"Position robot (grille): ({start_row}, {start_col})")
            self.get_logger().info(f"Goal (grille): ({goal_row}, {goal_col})")
            self.get_logger().info(f"Calcul du chemin...")
            
            # Vérifier que start et goal sont libres
            if self.grid[start_row, start_col] == 1:
                self.get_logger().warn(f"Position de départ est un obstacle! Cherche position libre...")
                # Prendre la première cellule libre trouvée
                if len(self.free_cell) > 0:
                    start_row, start_col = self.free_cell[0]
                    self.get_logger().info(f"Nouvelle position de départ (grille): ({start_row}, {start_col})")
            
            if self.grid[goal_row, goal_col] == 1:
                self.get_logger().warn(f"Goal est un obstacle! Cherche position libre proche...")
                # Prendre la première cellule libre trouvée
                if len(self.free_cell) > 0:
                    goal_row, goal_col = self.free_cell[0]
                    self.get_logger().info(f"Nouveau goal (grille): ({goal_row}, {goal_col})")
            
            self.path = astar(self.grid, (start_row, start_col), (goal_row, goal_col))
            self.path_computed = True

            if not self.path:
                self.get_logger().warn("Aucun chemin trouvé!")
                return
            else:
                first_world = self.map_to_world(self.path[0][0], self.path[0][1])
                last_world = self.map_to_world(self.path[-1][0], self.path[-1][1])
                self.get_logger().info(f"✓ Chemin trouvé ({len(self.path)} waypoints)")
                self.get_logger().info(f"  Début (monde): {first_world}")
                self.get_logger().info(f"  Fin (monde): {last_world}")

        # Récupérer la pose du robot (mise à jour)
        rx = self.pose.position.x
        ry = self.pose.position.y
        current_yaw = self.get_yaw(self.pose.orientation)

        # Vérifier s'il y a des waypoints restants
        if not self.path:
            self.get_logger().info("🎉 BUT ATTEINT!")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_publisher.publish(twist)
            return

        # Obtenir le prochain waypoint
        next_wp = self.path[0]
        wx, wy = self.map_to_world(next_wp[0], next_wp[1])

        # Calculer la distance et l'angle vers le waypoint
        dx = wx - rx
        dy = wy - ry
        dist = math.hypot(dx, dy)

        # Angle désiré vers le waypoint
        desired_yaw = math.atan2(dy, dx)

        # Erreur angulaire
        err_yaw = desired_yaw - current_yaw
        err_yaw = self._normalize_angle(err_yaw)

        # Gains du contrôleur
        k_rho = 2.0      # gain pour la distance
        k_alpha = 2.5    # gain pour l'angle
        k_beta = -1.0    # gain pour correction

        # Commandes de vitesse
        linear_speed = k_rho * dist

        # Limiter la vitesse linéaire maximale
        MAX_LINEAR_SPEED = 0.1
        linear_speed = max(-MAX_LINEAR_SPEED, min(MAX_LINEAR_SPEED, linear_speed))

        # Vitesse angulaire
        angular_speed = k_alpha * err_yaw + k_beta * err_yaw

        # Limiter la vitesse angulaire
        MAX_ANGULAR_SPEED = 0.5
        angular_speed = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angular_speed))

        # Publier les commandes
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_publisher.publish(twist)

        # Vérifier si le waypoint est atteint
        WAYPOINT_THRESHOLD = 0.15
        if dist < WAYPOINT_THRESHOLD:
            self.path.pop(0)
            remaining = len(self.path)
            if remaining > 0:
                self.get_logger().info(f"✓ Waypoint atteint! {remaining} restants")
            else:
                self.get_logger().info("BUT ATTEINT!")
        
        self.iteration_count += 1

        
def main(args=None):
    rclpy.init(args=args)
    try:
        navigator = Trajectoire(
            goal_x=1.0, 
            goal_y=2.0)

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