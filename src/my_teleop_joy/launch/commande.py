from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    return LaunchDescription([
        # Node(
        #     package='my_teleop_joy',
        #     executable='Trajectoire',
        #     name='Trajectoire',
        #     parameters=[{'use_sim_time': use_sim_time}]
        # )
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
        Node(
            package='my_teleop_joy',
            executable='goal_node',
            name='goal_publisher',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        )
    ])
    
        
    
