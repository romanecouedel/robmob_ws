#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

class ModeSwitcher(Node):
    """
    Nœud qui bascule entre mode téléopération et navigation autonome.
    
    Bouton de switch : Bouton (généralement 'A' sur manette Xbox, 'X' sur PS4)
    
    Modes:
    - TELEOP : Les commandes du joystick passent directement au robot
    - AUTO : La navigation autonome contrôle le robot
    """
    
    def __init__(self):
        super().__init__('mode_switcher')
        
        # État initial : mode téléopération
        self.current_mode = 'TELEOP'  # 'TELEOP' ou 'AUTO'
        self.last_button_state = 0
        
        # Paramètre : quel bouton pour switcher (défaut : bouton 0)
        self.declare_parameter('switch_button', 0)
        self.switch_button = self.get_parameter('switch_button').value
        
        # Subscribers
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        
        # Publishers
        self.teleop_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mode_pub = self.create_publisher(String, '/current_mode', 10)
        self.teleop_enable_pub = self.create_publisher(Bool, '/teleop/enable', 10)
        self.nav_enable_pub = self.create_publisher(Bool, '/nav/enable', 10)
        
        
        # Timer pour publier le mode actuel
        self.mode_timer = self.create_timer(0.5, self.publish_mode)
        
        self.get_logger().info(f'Mode Switcher initialisé en mode {self.current_mode}')
        self.get_logger().info(f'Appuyez sur le bouton {self.switch_button} pour basculer')
    
    def joy_callback(self, msg):
        """Détecte l'appui sur le bouton de switch"""
        # Convertir les axes du joystick en commandes de mouvement
        twist = Twist()
        twist.linear.x = msg.axes[1]
        twist.angular.z = msg.axes[0]
        if self.current_mode == 'TELEOP':
            self.teleop_cmd_pub.publish(twist)
        
        if len(msg.buttons) > self.switch_button:
            current_button = msg.buttons[self.switch_button]
            
            # Détection du front montant (bouton vient d'être pressé)
            if current_button == 1 and self.last_button_state == 0:
                self.toggle_mode()
            
            self.last_button_state = current_button
    
    def toggle_mode(self):
        """Bascule entre les modes"""
        if self.current_mode == 'TELEOP':
            self.current_mode = 'AUTO'
            self.get_logger().info(' PASSAGE EN MODE AUTONOME')
        else:
            self.current_mode = 'TELEOP'
            self.get_logger().info(' PASSAGE EN MODE TÉLÉOPÉRATION')
        
        # Publier l'état d'activation aux nœuds
        self.publish_enable_states()
    
    def publish_mode(self):
        """Publie le mode actuel"""
        mode_msg = String()
        mode_msg.data = self.current_mode
        self.mode_pub.publish(mode_msg)
    
    def publish_enable_states(self):
        """Publie l'état d'activation aux différents modules"""
        teleop_enable = Bool()
        nav_enable = Bool()
        
        if self.current_mode == 'TELEOP':
            teleop_enable.data = True
            nav_enable.data = False
        else:
            teleop_enable.data = False
            nav_enable.data = True
        
        self.teleop_enable_pub.publish(teleop_enable)
        self.nav_enable_pub.publish(nav_enable)

def main(args=None):
    rclpy.init(args=args)
    node = ModeSwitcher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()