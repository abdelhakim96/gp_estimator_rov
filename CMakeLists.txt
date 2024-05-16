cmake_minimum_required(VERSION 3.5)
project(gp_estimator)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
#find_package(px4_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)



include_directories(
  include
  )



# find GPKit
add_subdirectory(libraries/GPKit GPKit_build)


add_executable(gp_node src/gp_node.cpp)
target_include_directories(gp_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/libraries/GPKit/include
)

target_link_libraries(gp_node gpkit ${CMAKE_CURRENT_SOURCE_DIR}/libraries/GPKit/build/libgpkit.a)
ament_target_dependencies(gp_node rclcpp std_msgs geometry_msgs nav_msgs  )

install(TARGETS gp_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

