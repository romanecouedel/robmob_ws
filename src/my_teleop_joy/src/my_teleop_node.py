#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node 

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist 

class Subscriber(Node):

    def __init__(self):
        super().__init__('sub')
        self.sub = self.create_subscription(
            Joy,
            '/joy',
            self.listener_callback,
            10)
        self.pub = self.create_publisher(Twist,'/cmd_vel',10)
        

        
    def listener_callback(self, msg):
       twist=Twist()
       twist.linear.x=msg.axes[1]
       twist.angular.z=msg.axes[0]
       self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)

    sub_1 = Subscriber()
    rclpy.spin(sub_1)
    

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    sub_1.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()