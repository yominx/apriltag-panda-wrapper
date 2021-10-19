cmake_minimum_required(VERSION 3.1.0)

project(apriltag-panda-wrapper)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
find_package(catkin REQUIRED 
  COMPONENTS
	geometry_msgs
	roscpp
	tf
)
## Find system libraries
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(apriltag REQUIRED)
find_package(realsense2 REQUIRED)

# catkin specific configuration
catkin_package(
  INCLUDE_DIRS
  LIBRARIES 
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
	tf
  DEPENDS
)

include_directories(
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR} 
	${realsense2_INCLUDE_DIR}
	${apriltag_INCLUDE_DIR}
)

set(DEPENDENCIES 
	realsense2 
	${catkin_LIBRARIES}
	${OpenCV_LIBS} 
	apriltag::apriltag
)

## Declare cpp executables
# add_executable(calibration_node get_base_to_cam.cpp)
# target_link_libraries(calibration_node ${DEPENDENCIES})

add_executable(marker_publisher_node marker_publisher_node.cpp)
target_link_libraries(marker_publisher_node ${DEPENDENCIES})


# set_target_properties (${CMAKE_PROJECT_NAME} PROPERTIES
#	FOLDER "Examples"
# )

# install(
# 	TARGETS
# 	${CMAKE_PROJECT_NAME}

# 	RUNTIME DESTINATION
# 	${CMAKE_INSTALL_PREFIX}/bin
# )
