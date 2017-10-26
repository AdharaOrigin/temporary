# PROJECT
cmake_minimum_required(VERSION 3.7)
project(eye_control)

# FLAGS
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse2 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse2 -Wall")
set(CMAKE_BUILD_TYPE Release ... FORCE)

# ADD PACKAGES
# Where to find CMake modules and OpenCV
set(OpenCV_DIR "C:\\opencv320\\mingw-build\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# INCLUDES
include(lib/dlib/cmake)

# GROUPING SOURCE FILES
file(GLOB SOURCES_ELLIPTIC_DETECTION "spikes/elliptic_detection/*.h" "spikes/elliptic_detection/*.cpp")
file(GLOB SOURCES_RANSAC "spikes/ransac/*.h" "spikes/ransac/*.cpp")
file(GLOB SOURCES_POG "src/*.h" "src/*.cpp")
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_videoio opencv_photo opencv_objdetect)


# EXECUTABLES
# add_executable(elliptic_detection ${SOURCES_ELLIPTIC_DETECTION})
# TARGET_LINK_LIBRARIES(elliptic_detection ${OpenCV_LIBS} dlib::dlib)

add_executable(ransac ${SOURCES_RANSAC})
TARGET_LINK_LIBRARIES(ransac ${OpenCV_LIBS})

add_executable(point_of_gaze ${SOURCES_POG})
TARGET_LINK_LIBRARIES(point_of_gaze ${OpenCV_LIBS} dlib::dlib)