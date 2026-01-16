from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    return LaunchDescription([

        # 1. Map Manager
        Node(
            package='navigation',
            executable='pathmanager.py',
            name='path_manager',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),

        # 2. Trajectory Planner
        Node(
            package='navigation',
            executable='trajplannig.py',
            name='trajectory_planner',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        )
    ])
