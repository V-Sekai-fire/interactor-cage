# Native (host) build of cage.elf's code: guest/cage, bhc13 and the kernels'
# cpp emits, guest/drape's L-BFGS-B, plus curvenet_core for the scripted pen
# (cmake/curvenet.cmake, the curvenet stage's libraries). The flat control of
# gates/9-cage: if something fails here, the sandbox is not to blame.
#
#   tests/cage_native/build.sh
cmake_minimum_required(VERSION 3.16)
project(cage_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/curvenet.cmake)
set(R ${CMAKE_CURRENT_LIST_DIR}/../..)

add_executable(cage_native
	${CMAKE_CURRENT_LIST_DIR}/main.cpp
	${R}/guest/cage/cage_api.cpp
	${R}/guest/cage/cage_fit.cpp
	${R}/guest/cage/cage_net.cpp
	${R}/guest/cage/checks.cpp
	${R}/guest/cage/fit_driver.cpp
	${R}/guest/cage/fixtures.cpp
	${R}/guest/common/deform/bhc13/bhc13.cpp
	${R}/guest/common/deform/cage_kernels.cpp
	${R}/guest/drape/lbfgsb.cpp
	${R}/guest/drape/vec_cpu.cpp
)
# The guest builds these -ffp-contract=off (CMakeLists.txt); so does this.
target_compile_options(cage_native PRIVATE -ffp-contract=off -Wall -Wno-unused-function)
target_include_directories(cage_native PRIVATE ${R}/guest ${R}/guest/cage ${R}/guest/drape ${R}/guest/common/deform
	${R}/guest/avbd/slang-rt ${R}/guest/curvenet)
target_link_libraries(cage_native PRIVATE curvenet_core pmp_subset)
