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
        ('share/' + package_name +'/launch',['launch/launch.py']),
        ('share/' + package_name +'/rviz',['rviz/robmob.rviz']),
        ('share/' + package_name +'/launch',['launch/rviz.launch.py']),
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
            'my_teleop_node = my_teleop_joy.my_teleop_node:main'
        ],
    },
)
