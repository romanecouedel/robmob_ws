from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    pkg_share = get_package_share_directory("my_teleop_joy")
    rviz_config = os.path.join(pkg_share, "rviz", "robmob.rviz")

    tb3_pkg = get_package_share_directory("turtlebot3_gazebo")
    tb3_dqn_launch = os.path.join(tb3_pkg, "launch", "turtlebot3_dqn_stage4.launch.py")

    slam_pkg = get_package_share_directory("slam_toolbox")
    slam_launch = os.path.join(slam_pkg, "launch", "online_async_launch.py")

    return LaunchDescription([
    
        # SLAM: Cartographie en temps réel
        IncludeLaunchDescription(PythonLaunchDescriptionSource(tb3_dqn_launch),launch_arguments={"use_sim_time": use_sim_time}.items(),),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),
        
        
        Node(
            package='joy_linux',
            executable='joy_linux_node',
        ),
        
        Node(
            package='my_teleop_joy',
            executable='my_teleop_node',
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": use_sim_time}],
            output="screen",
        ),
        
          # 2. Map Manager - gère la map et calcule les chemins
        Node(
            package='my_teleop_joy',
            executable='path_manager_node',
            name='path_manager',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),        # 3. Trajectory Planner - expose /set_goal et /start_navigation
        Node(
            package='my_teleop_joy',
            executable='trajectory_planner_node',
            name='trajectory_planner',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
        # 4. Exploration Node - exploration autonome
        Node(
            package='my_teleop_joy',
            executable='exploration_node',
            name='exploration_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),

    ])