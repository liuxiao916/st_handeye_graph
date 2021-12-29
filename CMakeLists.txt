cmake_minimum_required(VERSION 3.10)
project(st_handeye)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

#add_compile_options(-std=c++11)
# C++14 support
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BOOST_ROOT /home/liuxiao/Boost/include)
set(BOOST_LIBRARYDIR /home/liuxiao/Boost/lib)
set(Boost_NO_SYSTEM_PATHS ON)

find_package(G2O REQUIRED)
find_package(VISP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.70.0 EXACT REQUIRED)
find_package(Glog REQUIRED)
find_package(CCTag CONFIG REQUIRED)



if(Boost_FOUND)
  MESSAGE( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
  MESSAGE( STATUS "Boost_LIBRARIES = ${Boost_LIBRARY_DIRS}.")
  MESSAGE( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")
  MESSAGE( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}.")
  
else()
  MESSAGE( WARNING "Not found Boost")
endif()

include_directories(
	include
        ${G2O_INCLUDE_DIRS}
        ${VISP_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${GLOG_INCLUDE_DIRS}
)

link_directories(${G2O_LIBRARY_DIRS}  ${GLOG_LIBRARY})

###########
## Build ##
###########


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(st_handeye_graph
  src/st_handeye_graph.cpp
)
target_link_libraries(st_handeye_graph
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${G2O_TYPES_DATA}
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_SOLVER_PCG}
  ${G2O_SOLVER_CSPARSE}
  ${G2O_SOLVER_CHOLMOD}
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_SLAM3D_ADDONS}
 ${GLOG_LIBRARY}
)

add_library(st_handeye_graph_xray
  src/st_handeye_graph_xray.cpp
)
target_link_libraries(st_handeye_graph_xray
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${G2O_TYPES_DATA}
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_SOLVER_PCG}
  ${G2O_SOLVER_CSPARSE}
  ${G2O_SOLVER_CHOLMOD}
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_SLAM3D_ADDONS}
 ${GLOG_LIBRARY}
)

add_library(st_handeye_visp
  src/st_handeye_visp.cpp
)
target_link_libraries(st_handeye_visp
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${VISP_LIBRARIES}
 glog::glog
)

# handeye_simulation
add_executable(handeye_simulation
  src/handeye_simulation.cpp
)
target_link_libraries(handeye_simulation
  st_handeye_graph
  st_handeye_visp
 glog::glog
)
add_dependencies(handeye_simulation
  st_handeye_graph
  st_handeye_visp
)

# handeye_simulation_xray
add_executable(handeye_simulation_xray
  src/handeye_simulation_xray.cpp
)
target_link_libraries(handeye_simulation_xray
    st_handeye_graph_xray
glog::glog
)
add_dependencies(handeye_simulation_xray
  st_handeye_graph_xray
)

# calibrate
add_executable(calibrate
  src/calibrate.cpp
)
target_link_libraries(calibrate
    st_handeye_graph
    st_handeye_visp
)
add_dependencies(calibrate
  st_handeye_graph
  st_handeye_visp
)

#cctag
add_executable(cctag
  src/cctag.cpp
)

target_link_libraries(cctag
    ${OpenCV_LIBRARIES}
    CCTag::CCTag 
)


# calibrate_cctag
add_executable(calibrate_cctag
  src/calibrate_cctag.cpp
)
target_link_libraries(calibrate_cctag
    st_handeye_graph
    st_handeye_visp
    CCTag::CCTag 
)
add_dependencies(calibrate_cctag
  st_handeye_graph
  st_handeye_visp
)

