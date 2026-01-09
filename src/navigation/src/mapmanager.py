#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from my_teleop_joy.srv import SetGoal, ComputePath
from nav_msgs.msg import OccupancyGrid
import numpy as np
import heapq

# ===== A* ALGORITHM =====
def heuristic(a, b):
    """Manhattan distance"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal, max_iterations=50000):
    """A* pathfinding"""
    if grid[goal[0], goal[1]] != 0:
        return []
    if grid[start[0], start[1]] != 0:
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
    
    return []

# ===== MAP MANAGER NODE =====
class MapManager(Node):
    def __init__(self):
        super().__init__('map_manager')
        
        # Souscription à la map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        
        # Service pour calculer le chemin
        self.compute_path_srv = self.create_service(
            ComputePath,
            '/computepath',
            self.compute_path_callback)
        
        self.grid = None
        self.map_width = None
        self.map_height = None
        self.msg_grid = None
        self.map_received = False
        
        self.get_logger().info("MapManager initialized - waiting for /map")

    def map_callback(self, msg: OccupancyGrid):
        """Recevoir et traiter la map"""
        if self.map_received:
            return
        
        self.get_logger().info(f"Map reçue: {msg.info.width}x{msg.info.height}")
        
        self.msg_grid = msg
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        
        # Convertir OccupancyGrid en numpy array
        grid_data = np.array(msg.data, dtype=np.int8).reshape(
            (self.map_height, self.map_width))
        
        # Normaliser: 1 = obstacle, 0 = libre
        self.grid = np.where(grid_data >= 50, 1, 0).astype(np.uint8)
        self.map_received = True
        
        self.get_logger().info("✓ Map prête")

    def world_to_map(self, x, y):
        """Convertir coords monde → grille"""
        origin_x = self.msg_grid.info.origin.position.x
        origin_y = self.msg_grid.info.origin.position.y
        res = self.msg_grid.info.resolution
        
        col = int((x - origin_x) / res)
        row = int((y - origin_y) / res)
        
        col = max(0, min(col, self.map_width - 1))
        row = max(0, min(row, self.map_height - 1))
        
        return (row, col)

    def map_to_world(self, row, col):
        """Convertir grille → coords monde"""
        origin_x = self.msg_grid.info.origin.position.x
        origin_y = self.msg_grid.info.origin.position.y
        res = self.msg_grid.info.resolution
        
        x = origin_x + col * res
        y = origin_y + row * res
        
        return (x, y)

    def compute_path_callback(self, request, response):
        """Service: calculer le chemin"""
        
        if not self.map_received:
            response.success = False
            response.message = "Map not received yet"
            return response
        
        try:
            # Convertir coords monde → grille
            start_row, start_col = self.world_to_map(request.start_x, request.start_y)
            goal_row, goal_col = self.world_to_map(request.goal_x, request.goal_y)
            
            self.get_logger().info(f"Computing path from ({request.start_x:.2f}, {request.start_y:.2f}) to ({request.goal_x:.2f}, {request.goal_y:.2f})")
            
            # Vérifier que les positions sont libres
            if self.grid[start_row, start_col] == 1:
                free_cells = np.argwhere(self.grid == 0)
                if len(free_cells) > 0:
                    start_row, start_col = free_cells[0]
                    self.get_logger().warn(f"Start was obstacle, adjusted to ({start_row}, {start_col})")
            
            if self.grid[goal_row, goal_col] == 1:
                free_cells = np.argwhere(self.grid == 0)
                if len(free_cells) > 0:
                    goal_row, goal_col = free_cells[0]
                    self.get_logger().warn(f"Goal was obstacle, adjusted to ({goal_row}, {goal_col})")
            
            # Calculer le chemin
            path = astar(self.grid, (start_row, start_col), (goal_row, goal_col))
            
            if not path:
                response.success = False
                response.message = "No path found"
                self.get_logger().error("No path found!")
                return response
            
            # Convertir le chemin en coords monde
            path_x = []
            path_y = []
            for row, col in path:
                x, y = self.map_to_world(row, col)
                path_x.append(int(x * 100))  # Stocker en cm
                path_y.append(int(y * 100))
            
            response.success = True
            response.path_x = path_x
            response.path_y = path_y
            response.message = f"Path computed with {len(path)} waypoints"
            
            self.get_logger().info(f"✓ Path computed: {len(path)} waypoints")
            
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f"Error: {e}")
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MapManager()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()