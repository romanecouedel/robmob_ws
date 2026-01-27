cmake_minimum_required(VERSION 3.5)
project(mission_msgs)

set(LIBRARY_NAME "mission_msgs")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${LIBRARY_NAME}
  "srv/MissionCancel.srv"
  "srv/MissionRequest.srv"
  "msg/Trajectory.msg"
  "msg/TrajectoryPoint.msg"
  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
  ADD_LINTER_TESTS
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
