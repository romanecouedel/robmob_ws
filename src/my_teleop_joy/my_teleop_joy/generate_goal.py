
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

class GenerateGoal(Node):
    def __init__(self):
        super().__init__('generate_goal')
        self.goal_pub = self.create_publisher(Pose, '/goal', 10)
        
    def publish_goal(self):
        goal_msg = Pose()        
        goal_msg.position.x = 1.0
        goal_msg.position.y = 2.0
        goal_msg.position.z = 0.0
        # id en orientation 
        goal_msg.orientation.x = 0.0
        goal_msg.orientation.y = 0.0
        goal_msg.orientation.z = 0.0
        goal_msg.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)
        
def main(args=None):
    rclpy.init(args=args)
    generate_goal_node = GenerateGoal()
    
    try:
        while rclpy.ok():
            generate_goal_node.publish_goal()
            rclpy.spin_once(generate_goal_node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        generate_goal_node.destroy_node()
        rclpy.shutdown()