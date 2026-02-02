#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

class ModeSwitcher(Node):
    """
    Nœud qui bascule entre 3 modes : téléopération, navigation autonome, et exploration.
    
    Boutons:
    - Bouton principal (ex: A/X) : Cycle entre les modes TELEOP → AUTO → EXPLORATION → TELEOP
    - OU boutons séparés (optionnel):
      * Bouton 0 : TELEOP ↔ AUTO
      * Bouton 1 : EXPLORATION
    
    Modes:
    - TELEOP : Les commandes du joystick passent directement au robot
    - AUTO : Navigation autonome vers un goal (suit /computed_path)
    - EXPLORATION : Exploration autonome de l'environnement
    """
    
    def __init__(self):
        super().__init__('mode_switcher')
        
        # État initial : mode téléopération
        self.current_mode = 'TELEOP'  # 'TELEOP', 'AUTO', ou 'EXPLORATION'
        self.last_button_state = [0] * 12  # Garder l'état de plusieurs boutons
        
        # Paramètres : boutons de switch
        self.declare_parameter('mode_cycle_button', 0)  # Bouton pour cycler entre les modes
        self.declare_parameter('exploration_button', 1)  # Bouton direct pour exploration
        
        self.mode_cycle_button = self.get_parameter('mode_cycle_button').value
        self.exploration_button = self.get_parameter('exploration_button').value
        
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
        
        # Publishers d'activation pour chaque mode
        self.teleop_enable_pub = self.create_publisher(Bool, '/teleop/enable', 10)
        self.nav_enable_pub = self.create_publisher(Bool, '/nav/enable', 10)
        self.exploration_enable_pub = self.create_publisher(Bool, '/exploration/enable', 10)
        
        # Timer pour publier le mode actuel
        self.mode_timer = self.create_timer(0.5, self.publish_mode)
        
        self.get_logger().info('╔════════════════════════════════════════════════╗')
        self.get_logger().info('║      MODE SWITCHER - 3 MODES DISPONIBLES      ║')
        self.get_logger().info('╠════════════════════════════════════════════════╣')
        self.get_logger().info(f'║  Mode actuel: {self.current_mode:<32} ║')
        self.get_logger().info('║                                                ║')
        self.get_logger().info(f'║  Bouton {self.mode_cycle_button}: Cycler modes (TELEOP→AUTO→EXPLO) ║')
        self.get_logger().info('║                                                ║')
        self.get_logger().info('║  🕹️  TELEOP      : Contrôle manuel            ║')
        self.get_logger().info('║  🎯 AUTO         : Navigation vers goal       ║')
        self.get_logger().info('║  🔍 EXPLORATION  : Exploration autonome        ║')
        self.get_logger().info('╚════════════════════════════════════════════════╝')
    
    def joy_callback(self, msg):
        """Détecte l'appui sur les boutons de switch"""
        
        # En mode TELEOP, transmettre les commandes du joystick
        if self.current_mode == 'TELEOP':
            twist = Twist()
            twist.linear.x = msg.axes[1]  # Stick gauche vertical
            twist.angular.z = msg.axes[0]  # Stick gauche horizontal
            self.teleop_cmd_pub.publish(twist)
        
        # Vérifier les boutons
        if len(msg.buttons) > max(self.mode_cycle_button, self.exploration_button):
            
            # Bouton de cycle des modes
            if (msg.buttons[self.mode_cycle_button] == 1 and 
                self.last_button_state[self.mode_cycle_button] == 0):
                self.cycle_mode()
            
            # Mettre à jour l'état des boutons
            self.last_button_state = list(msg.buttons)
    
    def cycle_mode(self):
        """Cycle entre les 3 modes dans l'ordre: TELEOP → AUTO → EXPLORATION → TELEOP"""
        if self.current_mode == 'TELEOP':
            self.set_mode('AUTO')
        elif self.current_mode == 'AUTO':
            self.set_mode('EXPLORATION')
        else:  # EXPLORATION
            self.set_mode('TELEOP')
    
    def set_mode(self, new_mode):
        """Définir le mode actuel et publier les états"""
        old_mode = self.current_mode
        self.current_mode = new_mode
        
        # Logs avec emojis
        mode_icons = {
            'TELEOP': '🕹️ ',
            'AUTO': '🎯',
            'EXPLORATION': '🔍'
        }
        
        self.get_logger().info('═' * 60)
        self.get_logger().info(
            f'{mode_icons[old_mode]} {old_mode} → {mode_icons[new_mode]} {new_mode}'
        )
        
        # Détails sur le mode
        if new_mode == 'TELEOP':
            self.get_logger().info('   Contrôle manuel activé - utilisez le joystick')
        elif new_mode == 'AUTO':
            self.get_logger().info('   Navigation autonome - en attente de goal (/goal_pose)')
            self.get_logger().info('   Placez un goal dans RViz (2D Goal Pose)')
        elif new_mode == 'EXPLORATION':
            self.get_logger().info('   Exploration autonome activée')
            self.get_logger().info('   Le robot va explorer l\'environnement')
        
        self.get_logger().info('═' * 60)
        
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
        exploration_enable = Bool()
        
        if self.current_mode == 'TELEOP':
            teleop_enable.data = True
            nav_enable.data = False
            exploration_enable.data = False
        elif self.current_mode == 'AUTO':
            teleop_enable.data = False
            nav_enable.data = True
            exploration_enable.data = False
        else:  # EXPLORATION
            teleop_enable.data = False
            nav_enable.data = False
            exploration_enable.data = True
        
        self.teleop_enable_pub.publish(teleop_enable)
        self.nav_enable_pub.publish(nav_enable)
        self.exploration_enable_pub.publish(exploration_enable)
        
        # Log de debug
        self.get_logger().debug(
            f'États publiés: teleop={teleop_enable.data}, '
            f'nav={nav_enable.data}, exploration={exploration_enable.data}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = ModeSwitcher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n👋 Arrêt du mode switcher')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()