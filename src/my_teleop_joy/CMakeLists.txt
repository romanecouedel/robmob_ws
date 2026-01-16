cmake_minimum_required(VERSION 3.8)
project(my_teleop_joy)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Générer les interfaces de services
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetGoal.srv"
  "srv/ComputePath.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

# # Installer les fichiers Python du package
# ament_python_install_package(my_teleop_joy)

# Installer les nodes Python exécutables
install(PROGRAMS
  my_teleop_joy/pathmanager.py
  my_teleop_joy/trajplannig.py
  my_teleop_joy/my_teleop_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Installer les fichiers de ressources
install(DIRECTORY
  launch
  params
  rviz
  srv
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpylint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()