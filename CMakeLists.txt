cmake_minimum_required(VERSION 3.1.0)

project(mark_detector)

find_package(OpenCV REQUIRED)
find_package(Franka REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
set(DEPENDENCIES realsense2 Franka::Franka apriltag ${OpenCV_LIBS})

add_executable(${CMAKE_PROJECT_NAME} get_base_to_cam.cpp apriltag.hpp)
target_link_libraries(${CMAKE_PROJECT_NAME} ${DEPENDENCIES})
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 11)

# set_target_properties (${CMAKE_PROJECT_NAME} PROPERTIES
#	FOLDER "Examples"
# )

install(
	TARGETS
	${CMAKE_PROJECT_NAME}

	RUNTIME DESTINATION
	${CMAKE_INSTALL_PREFIX}/bin
)