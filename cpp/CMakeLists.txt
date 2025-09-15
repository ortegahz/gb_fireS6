cmake_minimum_required(VERSION 3.14)
project(FireDetectorCpp)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV 4.x (请确保已安装并配置好环境)
find_package(OpenCV 4 REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV 4 not found.")
endif ()

# 添加头文件目录和链接库
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件及其源文件
add_executable(fire_detector_main
        main.cpp
        FireDetector.cpp
)

# 链接 OpenCV 库
target_link_libraries(fire_detector_main ${OpenCV_LIBS})