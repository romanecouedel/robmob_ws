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

        # ---------------- PARAMÈTRES ROBOT ----------------
        self.robot_radius = 0.22       # m
        self.inflation_margin = 0.05    # m
        # --------------------------------------------------

        # Configuration TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos_map = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos_map
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10
        )

        self.path_pub = self.create_publisher(Path, '/computed_path', 10)
        self.return_path_pub = self.create_publisher(Path, '/return_path', 10)

        self.map_received = False
        self.grid = None
        self.grid_inflated = None
        self.msg_grid = None
        self.map_width = None
        self.map_height = None
        self.free_cell = None

        self.goal_x = goal_world[0]
        self.goal_y = goal_world[1]
        self.path_computed = False
        self.last_robot_position = None

        self.get_logger().info("PathManager initialized - waiting for /map")

        while not self.map_received:
            rclpy.spin_once(self, timeout_sec=1.0)

        self.get_logger().info("✓ Map reçue - prêt à planifier")

    # ======================================================
    # TF
    # ======================================================
    def get_robot_pose_from_tf(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time()
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y
            )
        except Exception as e:
            self.get_logger().warn(f"TF indisponible: {e}")
            return None

    # ======================================================
    # MAP
    # ======================================================
    def map_callback(self, msg: OccupancyGrid):
        if self.map_received:
            return

        self.msg_grid = msg
        self.map_width = msg.info.width
        self.map_height = msg.info.height

        grid_data = np.array(msg.data, dtype=np.int8).reshape(
            (self.map_height, self.map_width)
        )

        self.grid = np.where(grid_data >= 50, 1, 0).astype(np.uint8)
        self.grid_inflated = self.inflate_grid(self.grid)

        self.free_cell = np.argwhere(self.grid_inflated == 0)

        if self.free_cell.size == 0:
            self.get_logger().error("Aucune cellule libre après inflation !")
            return

        self.map_received = True
        self.get_logger().info("✓ Map prête pour la planification")

    def inflate_grid(self, grid):
        res = self.msg_grid.info.resolution
        inflation_radius = self.robot_radius + self.inflation_margin
        inflation_cells = int(inflation_radius / res)

        inflated = grid.copy()
        obstacles = np.argwhere(grid == 1)

        for r, c in obstacles:
            for dr in range(-inflation_cells, inflation_cells + 1):
                for dc in range(-inflation_cells, inflation_cells + 1):
                    rr = r + dr
                    cc = c + dc
                    if (
                        0 <= rr < grid.shape[0]
                        and 0 <= cc < grid.shape[1]
                        and dr*dr + dc*dc <= inflation_cells**2
                    ):
                        inflated[rr, cc] = 1

        self.get_logger().info(
            f"Obstacles dilatés : {inflation_cells} cellules (~{inflation_radius:.2f} m)"
        )
        return inflated

    # ======================================================
    # COORDINATES
    # ======================================================
    def world_to_map(self, x, y):
        origin = self.msg_grid.info.origin.position
        res = self.msg_grid.info.resolution
        col = int((x - origin.x) / res)
        row = int((y - origin.y) / res)
        return (
            max(0, min(row, self.map_height - 1)),
            max(0, min(col, self.map_width - 1))
        )

    def map_to_world(self, row, col):
        origin = self.msg_grid.info.origin.position
        res = self.msg_grid.info.resolution
        return (
            origin.x + col * res,
            origin.y + row * res
        )

    # ======================================================
    # GOAL
    # ======================================================
    def goal_callback(self, msg: PoseStamped):
        if msg.header.frame_id != 'map':
            return

        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.path_computed = False

        if self.map_received:
            self.compute_path()

    # ======================================================
    # PATH COMPUTATION
    # ======================================================
    def compute_path(self):
        if self.path_computed:
            return

        robot_pose = self.get_robot_pose_from_tf()
        if robot_pose is None:
            return

        start_row, start_col = self.world_to_map(*robot_pose)
        goal_row, goal_col = self.world_to_map(self.goal_x, self.goal_y)

        planning_grid = self.grid_inflated

        if planning_grid[start_row, start_col] == 1:
            start_row, start_col = self.free_cell[0]

        if planning_grid[goal_row, goal_col] == 1:
            goal_row, goal_col = self.free_cell[0]

        path = PathManager.astar(
            planning_grid,
            (start_row, start_col),
            (goal_row, goal_col)
        )

        if not path:
            self.get_logger().error("Aucun chemin trouvé")
            return

        self.path_computed = True
        self.publish_path(path)

    # ======================================================
    # PATH PUBLISH
    # ======================================================
    def publish_path(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for r, c in path:
            x, y = self.map_to_world(r, c)
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    # ======================================================
    # A*
    # ======================================================
    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def astar(grid, start, goal):
        neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: PathManager.heuristic(start, goal)}
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            close_set.add(current)

            for i, j in neighbors:
                neighbor = (current[0] + i, current[1] + j)

                if not (0 <= neighbor[0] < grid.shape[0]
                        and 0 <= neighbor[1] < grid.shape[1]):
                    continue

                if grid[neighbor] == 1:
                    continue

                tentative_g = gscore[current] + 1

                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue

                if tentative_g < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + PathManager.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return []


def main(args=None):
    rclpy.init(args=args)
    node = PathManager(goal_world=(2.0, 2.0))
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
