import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import heapq

# -----------------------------
# Helper A* functions
# -----------------------------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal, max_iterations=50000):
    print("Goal:", goal, "valeur dans grid:", grid[goal[0], goal[1]])
    neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
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
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    print(f"A* timeout après {iterations} itérations - pas de chemin trouvé")
    return []

class CustomNavigator(Node):
    def __init__(self, map_path, goal_world=(0.0, 0.0), map_resolution=0.05):
        super().__init__('custom_navigator')

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # Load map
        try:
            map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            if map_img is None:
                raise RuntimeError(f"Impossible de charger la map: {map_path}")
            self.map_height, self.map_width = map_img.shape
            self.grid = np.where(map_img > 127, 0, 1)  # 1=obstacle, 0=libre
            free_cells = np.argwhere(self.grid == 0)
            occupied_cells = np.argwhere(self.grid == 1)
            print(f"Nombre de cellules libres: {len(free_cells)}")
            print(f"Nombre de cellules occupées: {len(occupied_cells)}")
        except Exception as e:
            self.get_logger().error(f"Erreur chargement map: {e}")
            raise

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

        # Robot state
        self.pose = None
        self.scan = None
        self.iteration_count = 0

        # Start position : première cellule libre
        free_cells = np.argwhere(self.grid == 0)
        start_cell = free_cells[0]
        self.start_map = (start_cell[0], start_cell[1])
        print(f"Start map choisi: {self.start_map}")

        # Goal : convertir goal_world vers la cellule libre la plus proche
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

        # Test conversions
        test_world = (0.5, 0.5)
        test_map = self.world_to_map(*test_world)
        test_world_back = self.map_to_world(*test_map)
        self.get_logger().info(f"Test conversion: {test_world} → {test_map} → {test_world_back}")

        # Timers
        self.create_timer(0.1, self.main_loop)
        self.create_timer(1.0, self.publish_map)

    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_callback(self, msg):
        self.pose = msg.pose.pose

    def scan_callback(self, msg):
        self.scan = msg

    # -----------------------------
    # Map publishing
    # -----------------------------
    def publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.map_msg)

    # -----------------------------
    # Coordinate transforms
    # -----------------------------
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

    def get_yaw(self, q):
        """Extraire yaw d'un quaternion"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    # -----------------------------
    # Obstacle avoidance amélioré
    # -----------------------------
    def avoid_obstacles(self):
        """Détection d'obstacles avec 3 secteurs"""
        if self.scan is None or len(self.scan.ranges) == 0:
            return 0.0
        
        ranges = np.array(self.scan.ranges)
        
        # Filtrer les valeurs invalides
        valid_ranges = ranges[(ranges > self.scan.range_min) & 
                              (ranges < self.scan.range_max) &
                              (~np.isnan(ranges)) &
                              (~np.isinf(ranges))]
        
        if len(valid_ranges) == 0:
            return 0.0
        
        n = len(valid_ranges)
        
        # Diviser en 3 secteurs
        left = np.min(valid_ranges[n//2:])        # À gauche
        front = np.min(valid_ranges[n//4:3*n//4]) # Devant
        right = np.min(valid_ranges[:n//2])       # À droite
        
        threshold = 0.5
        if front < threshold:
            # Obstacle devant : tourner vers le côté le plus dégagé
            if left > right:
                return 0.8  # Tourner à gauche
            else:
                return -0.8  # Tourner à droite
        
        return 0.0

    # -----------------------------
    # Main loop
    # -----------------------------
    def main_loop(self):
        if self.pose is None:
            return

        # Calculer le chemin si pas encore fait
        if not self.path:
            self.path = astar(self.grid, self.start_map, self.goal_map)
            if not self.path:
                self.get_logger().warn("Pas de chemin trouvé!")
                return
            else:
                self.get_logger().info(f"Chemin calculé: {len(self.path)} waypoints")
                self.get_logger().info(f"Start: {self.start_map}, Goal: {self.goal_map}")
                self.get_logger().info(f"Premier WP map: {self.path[0]}, Dernier: {self.path[-1]}")
                
                # Convertir pour afficher en monde
                first_world = self.map_to_world(self.path[0][0], self.path[0][1])
                last_world = self.map_to_world(self.path[-1][0], self.path[-1][1])
                self.get_logger().info(f"Premier WP monde: {first_world}, Dernier: {last_world}")

        # Suivre le chemin
        if self.path:
            next_wp = self.path[0]
            
          
            wx, wy = self.map_to_world(next_wp[0], next_wp[1])
            
            # DEBUG : logger position tous les 10 iterations
            if self.iteration_count % 10 == 0:
                self.get_logger().info(
                    f"Robot: ({self.pose.position.x:.2f}, {self.pose.position.y:.2f}) "
                    f"→ WP: ({wx:.2f}, {wy:.2f}), {len(self.path)} WP restants"
                )

            dx = wx - self.pose.position.x
            dy = wy - self.pose.position.y
            distance = math.hypot(dx, dy)
            
            angle_to_goal = math.atan2(dy, dx)
            yaw = self.get_yaw(self.pose.orientation)
            angle_error = angle_to_goal - yaw
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

            twist = Twist()
            
           
            twist.linear.x = min(0.3, max(0.0, distance * 0.5))
            twist.angular.z = max(-1.5, min(1.5, angle_error * 2.0))
            
            # Ajouter évitement obstacles
            twist.angular.z += self.avoid_obstacles() * 0.3
            twist.angular.z = max(-1.5, min(1.5, twist.angular.z))
            
            self.cmd_pub.publish(twist)

            # ✅ Threshold moins strict
            WAYPOINT_THRESHOLD = 0.15
            if distance < WAYPOINT_THRESHOLD:
                self.path.pop(0)
                if len(self.path) > 0:
                    self.get_logger().info(f"Waypoint atteint! {len(self.path)} restants")
                else:
                    self.get_logger().info("🎉 GOAL ATTEINT!")
            
            self.iteration_count += 1

# -----------------------------
# Main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    try:
        navigator = CustomNavigator(
            '/home/oualid/robmob_ws/map_inflated.pgm',
            goal_world=(0.1, 2.1)
        )
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        print("Arrêt par utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()