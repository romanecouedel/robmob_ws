from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # Récupérer les chemins des packages
    pkg_share = get_package_share_directory("navigation")
    
    # Chemins vers les autres packages
    tb3_pkg = get_package_share_directory("turtlebot3_gazebo")
    tb3_dqn_launch = os.path.join(tb3_pkg, "launch", "turtlebot3_dqn_stage2.launch.py")

    slam_pkg = get_package_share_directory("slam_toolbox")
    slam_launch = os.path.join(slam_pkg, "launch", "online_async_launch.py")
    
    # Fichiers de config du package navigation
    rviz_config = os.path.join(pkg_share, "rviz", "robmob.rviz")
    map_file = os.path.join(pkg_share, "params", "map_name.yaml")
    amcl_file = os.path.join(pkg_share, "params", "amcl.yaml")


    return LaunchDescription([

        # Lancer la simulation Gazebo + TurtleBot3
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tb3_dqn_launch),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),
        
        # Optionnel: SLAM (décommenter si nécessaire)
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(slam_launch),
        #     launch_arguments={"use_sim_time": use_sim_time}.items(),
        # ),

        # ===== NODES RVIZ ET JOYSTICK (existants) =====
        
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": use_sim_time}],
            output="screen",
        ),
        
        Node(
            package='joy_linux',
            executable='joy_linux_node',
            name='joy_linux_node',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        Node(
            package='my_teleop_joy',
            executable='my_teleop_node',
            name='my_teleop_node',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ===== NODES SERVICES DE NAVIGATION (nouveaux) =====

        # Map Manager: Gère la map et fournit le service /compute_path
        Node(
            package='navigation',
            executable='map_manager_node',
            name='map_manager_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),

        # Trajectory Planner: Expose /set_goal et /start_navigation
        Node(
            package='navigation',
            executable='trajectoire_planif',
            name='trajectoire_planif',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),

        # Optional: Robot Controller (si vous en avez besoin)
        Node(
             package='navigation',
             executable='boucle_commande',
             name='boucle_commande',
             parameters=[{'use_sim_time': use_sim_time}],
             output='screen',
         ),

        # ===== MAP SERVER ET AMCL (optionnel) =====

       Node(
           package='nav2_map_server',
           executable='map_server',
           name='map_server',
           output='screen',
           parameters=[{'use_sim_time': use_sim_time},
                       {'yaml_filename': map_file}]
       ),
    
       Node(
           package='nav2_amcl',
           executable='amcl',
           name='amcl',
           parameters=[amcl_file],
       ),
    
       Node(
           package='nav2_lifecycle_manager',
           executable='lifecycle_manager',
           name='lifecycle_manager_localization',
           output='screen',
           parameters=[{'use_sim_time': use_sim_time},
                       {'autostart': True},
                       {'node_names': ['amcl','map_server']}]
       ),
        
    ])