#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


class MapImageProcessor(Node):
    """
    Node pour traiter la map en temps réel avec OpenCV
    
    Applique des opérations morphologiques (dilatation, érosion, etc.)
    sur la map reçue et la publie sur un nouveau topic
    """
    
    def __init__(self):
        super().__init__('map_image_processor')
        
        # ✨ Paramètres de traitement d'image
        self.declare_parameter('input_topic', '/map')
        self.declare_parameter('output_topic', '/map_processed_cv')
        self.declare_parameter('kernel_size', 5)
        self.declare_parameter('erosion_iterations', 1)
        self.declare_parameter('dilation_iterations', 1)
        self.declare_parameter('morphology_operation', 'dilate')  # 'dilate', 'erode', 'open', 'close', 'gradient'
        self.declare_parameter('apply_blur', False)
        self.declare_parameter('blur_kernel_size', 5)
        self.declare_parameter('threshold_value', 127)
        
        # Récupérer les paramètres
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.kernel_size = self.get_parameter('kernel_size').value
        self.erosion_iterations = self.get_parameter('erosion_iterations').value
        self.dilation_iterations = self.get_parameter('dilation_iterations').value
        self.morphology_op = self.get_parameter('morphology_operation').value
        self.apply_blur = self.get_parameter('apply_blur').value
        self.blur_kernel_size = self.get_parameter('blur_kernel_size').value
        self.threshold_value = self.get_parameter('threshold_value').value
        
        # Créer l'élément structurant
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.kernel_size, self.kernel_size)
        )
        
        # Configurer le QoS
        qos_map = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # Subscriber et Publisher
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.input_topic,
            self.map_callback,
            qos_map
        )
        
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            self.output_topic,
            10
        )
        
        self.get_logger().info(f" Map Image Processor initialized")
        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Output topic: {self.output_topic}")
        self.get_logger().info(f"Kernel size: {self.kernel_size}x{self.kernel_size}")
        self.get_logger().info(f"Operation: {self.morphology_op}")

    def map_callback(self, msg: OccupancyGrid):
        """
        Callback pour traiter la map à chaque réception
        """
        self.get_logger().debug(f"Map reçue: {msg.info.width}x{msg.info.height}")
        
        #  Étape 1 : Convertir OccupancyGrid → numpy array (image)
        map_array = np.array(msg.data, dtype=np.uint8).reshape(
            (msg.info.height, msg.info.width)
        )
        
        #  Étape 2 : Appliquer le traitement d'image
        processed = self.process_image(map_array)
        
        #  Étape 3 : Convertir l'image traitée → OccupancyGrid
        processed_msg = self.create_occupancy_grid_msg(
            processed, 
            msg.info.width, 
            msg.info.height,
            msg.info.resolution,
            msg.info.origin,
            msg.header.frame_id
        )
        
        #  Étape 4 : Publier la map traitée
        self.map_pub.publish(processed_msg)

    def process_image(self, image):
        """
        Applique les opérations d'image traitées
        
        Opérations disponibles :
        - dilate   : Agrandit les obstacles
        - erode    : Réduit les obstacles
        - open     : Érosion puis dilatation (élimine bruit)
        - close    : Dilatation puis érosion (comble les trous)
        - gradient : Dilate - Erode (contours)
        """
        
        # ✨ Convertir en uint8 si nécessaire
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # ✨ Optionnel : Appliquer un flou (lisse l'image)
        if self.apply_blur:
            kernel_blur = (self.blur_kernel_size, self.blur_kernel_size)
            image = cv2.GaussianBlur(image, kernel_blur, 0)
            self.get_logger().debug("Blur appliqué")
        
        # ✨ Appliquer l'opération morphologique choisie
        if self.morphology_op == 'erode':
            processed = cv2.erode(
                image, 
                self.kernel, 
                iterations=self.dilation_iterations,
                borderType=cv2.BORDER_REPLICATE
            )
            self.get_logger().debug(f"Dilatation appliquée ({self.dilation_iterations} itérations)")
        
        elif self.morphology_op == 'erode':
            processed = cv2.erode(
                image, 
                self.kernel, 
                iterations=self.erosion_iterations,
                borderType=cv2.BORDER_REPLICATE
            )
            self.get_logger().debug(f"Érosion appliquée ({self.erosion_iterations} itérations)")
        
        elif self.morphology_op == 'open':
            # Ouverture = Érosion + Dilatation (élimine bruit)
            processed = cv2.morphologyEx(
                image, 
                cv2.MORPH_OPEN, 
                self.kernel,
                iterations=max(self.erosion_iterations, self.dilation_iterations)
            )
            self.get_logger().debug("Ouverture appliquée (Érosion + Dilatation)")
        
        elif self.morphology_op == 'close':
            # Fermeture = Dilatation + Érosion (comble les trous)
            processed = cv2.morphologyEx(
                image, 
                cv2.MORPH_CLOSE, 
                self.kernel,
                iterations=max(self.erosion_iterations, self.dilation_iterations)
            )
            self.get_logger().debug("Fermeture appliquée (Dilatation + Érosion)")
        
        elif self.morphology_op == 'gradient':
            # Gradient morphologique = Dilate - Erode (contours)
            processed = cv2.morphologyEx(
                image, 
                cv2.MORPH_GRADIENT, 
                self.kernel
            )
            self.get_logger().debug("Gradient morphologique appliqué")
        
        else:
            self.get_logger().warn(f"Opération inconnue: {self.morphology_op}")
            processed = image
        
        return processed

    def create_occupancy_grid_msg(self, image, width, height, resolution, origin, frame_id):
        """
        ✨ Convertir une image numpy en message OccupancyGrid
        """
        # Créer le message
        grid_msg = OccupancyGrid()
        
        # Copier le header et l'info
        grid_msg.header.frame_id = frame_id
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        
        grid_msg.info.width = width
        grid_msg.info.height = height
        grid_msg.info.resolution = resolution
        grid_msg.info.origin = origin
        
        # Convertir l'image en liste d'occupancy
        # Normaliser les valeurs : 0-100
        grid_data = (image.astype(np.float32) / 255.0 * 100.0).astype(np.int8)
        grid_msg.data = grid_data.flatten().tolist()
        
        return grid_msg


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MapImageProcessor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Arrêt par utilisateur")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()