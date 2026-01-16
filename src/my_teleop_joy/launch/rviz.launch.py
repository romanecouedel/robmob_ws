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
    tb3_dqn_launch = os.path.join(tb3_pkg, "launch", "turtlebot3_dqn_stage2.launch.py")

    slam_pkg = get_package_share_directory("slam_toolbox")
    
    map_file = os.path.join(get_package_share_directory(
        'my_teleop_joy'), 'params', 'map_name.yaml')
    
    amcl_file = os.path.join(get_package_share_directory(
        'my_teleop_joy'), 'params', 'amcl.yaml')
    

    return LaunchDescription([

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tb3_dqn_launch),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),
        
        # IncludeLaunchDescription(
        #      PythonLaunchDescriptionSource(slam_launch),
        #      launch_arguments={"use_sim_time": use_sim_time}.items(),
        #  ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": use_sim_time}],
            output="screen",
        ),
        
        # Node(
        #     package='joy_linux',
        #     executable='joy_linux_node',
        #     name='joy_linux_node',
        #     parameters=[{'use_sim_time': use_sim_time}],
        # ),

        # Node(
        #     package='my_teleop_joy',
        #     executable='my_teleop_node',
        #     name='my_teleop_node',
        #     parameters=[{'use_sim_time': use_sim_time}],
        # ),
      
        
     
        
        
    ])