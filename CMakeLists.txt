cmake_minimum_required(VERSION 2.8.3)
project(edge_imu_bias)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -g -fopenmp -Wall -mavx -ffast-math -std=c++11 -Ofast -pipe -march=native -mtune=native")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SSE_FLAGS} -g -fopenmp -Ofast -march=native -mtune=native -pipe -Wall")
SET(EIGEN_INCLUDE_DIR "/usr/include/eigen3")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs 
  message_generation
  camera_info_manager 
  dynamic_reconfigure
  cmake_modules
  tf
)

find_package(libvisensor 1.1.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)
find_package(Ceres REQUIRED)

generate_dynamic_reconfigure_options(cfg/Params.cfg)

catkin_package(
    DEPENDS
    CATKIN_DEPENDS
)

FILE(GLOB HEADER_FILES src/*.h* src/Tracking/*.h* )
FILE(GLOB SOURCE_FILES src/*.c* src/Tracking/*.c* )

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${libvisensor_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/
)

set(LIBS
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    ${CERES_LIBRARIES}
    ${libvisensor_LIBRARIES}
    X11
)

add_executable(slam main_live_odometry.cpp ${HEADER_FILES} ${SOURCE_FILES})
add_dependencies(slam ${PROJECT_NAME}_gencfg)
target_link_libraries(slam ${LIBS})

