#!/usr/bin/env python3

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
import heapq

# -----------------------------
# Helper A* functions
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
    
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
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
            
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            
            if grid[neighbor[0], neighbor[1]] != 0:
                continue
            
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

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

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
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_data = None
        self.pose = None
        self.scan = None
        self.iteration_count = 0
        self.map_received = False
        self.path_found = False  
        
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

        self.create_timer(0.5, self.cmd)


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
            self.get_logger().error(" Aucune cellule libre trouvée dans la map!")
            return
        
        self.map_received = True
        self.get_logger().info(f" Map prête pour la planification")


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
        
        # ===== PHASE 1: CALCULER LE CHEMIN (une seule fois) =====
        if not self.path_computed:
            rx = self.pose.position.x
            ry = self.pose.position.y
            
            goal_row, goal_col = self.world_to_map(self.goal_x, self.goal_y)
            start_row, start_col = self.world_to_map(rx, ry)

            self.get_logger().info(f"Position robot (monde): ({rx:.2f}, {ry:.2f})")
            self.get_logger().info(f"Position robot (grille): ({start_row}, {start_col})")
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
            self.path = astar(self.grid, (start_row, start_col), (goal_row, goal_col))
            self.path_computed = True

            # VERIFICATION IMPORTANTE: Si pas de chemin, arrêter
            if not self.path:
                self.get_logger().error(" AUCUN CHEMIN TROUVÉ! Robot arrêté.")
                self.path_found = False
                # Publier commande d'arrêt
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_publisher.publish(twist)
                return
            
            # Chemin trouvé avec succès
            self.path_found = True
            first_world = self.map_to_world(self.path[0][0], self.path[0][1])
            last_world = self.map_to_world(self.path[-1][0], self.path[-1][1])
            self.get_logger().info(f"  Chemin trouvé ({len(self.path)} waypoints)")
            self.get_logger().info(f"  Début (monde): {first_world}")
            self.get_logger().info(f"  Fin (monde): {last_world}")

        # ===== PHASE 2: SI CHEMIN NON TROUVÉ, NE RIEN FAIRE =====
        if not self.path_found:
            return

        # ===== PHASE 3: SUIVRE LE CHEMIN =====
        
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

        # Calculer distance et angle
        dx = wx - rx
        dy = wy - ry
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        err_yaw = self._normalize_angle(desired_yaw - current_yaw)

        # Contrôle proportionnel
        k_rho = 2.0
        k_alpha = 2.5
        k_beta = -1.0

        linear_speed = k_rho * dist
        MAX_LINEAR_SPEED = 0.1
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
                self.get_logger().info(f"✓ Waypoint atteint! {remaining} restants")
            else:
                self.get_logger().info("BUT ATTEINT!")
        
        self.iteration_count += 1

        
def main(args=None):
    rclpy.init(args=args)
    try:
        navigator = Trajectoire(goal_x=1.0, goal_y=2.0)
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
    
# -----------------------------
# Base-link -> odom  -> map