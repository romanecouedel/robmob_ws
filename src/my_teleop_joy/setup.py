from setuptools import find_packages, setup

package_name = 'my_teleop_joy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name +'/launch',['launch/commande.py']),
        ('share/' + package_name +'/launch',['launch/exploration_launch.py']),
        ('share/' + package_name +'/rviz',['rviz/robmob.rviz']),
        ('share/' + package_name +'/launch',['launch/rviz.launch.py']),
        ('share/' + package_name +'/launch',['launch/start_map_server_amcl.launch.py']),
        ('share/' + package_name +'/params',['params/amcl.yaml']),
        ('share/' + package_name +'/params',['params/map_name.yaml']),
        ('share/' + package_name +'/params',['params/slam_toolbox_params.yaml']),
        ('share/' + package_name +'/params',['params/map_name.pgm']),
        ('share/' + package_name +'/params',['params/map_inflated.pgm']),
        ('share/' + package_name +'/srv',['srv/SetGoal.srv']),
        ('share/' + package_name +'/srv',['srv/ComputePath.srv']),
        ('share/' + package_name +'/my_teleop_joy',['my_teleop_joy/traj1.py']),
        ('share/' + package_name +'/my_teleop_joy',['my_teleop_joy/pathmanager.py']),
        ('share/' + package_name +'/my_teleop_joy',['my_teleop_joy/trajplannig.py'])
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oualid',
    maintainer_email='oualidboudemagh22@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'my_teleop_node = my_teleop_joy.my_teleop_node:main',
            #'Trajectoire = my_teleop_joy.traj1:main',
            'path_manager_node = my_teleop_joy.pathmanager:main',
            'trajectory_planner_node = my_teleop_joy.trajplannig:main',
        ],
    },
)
