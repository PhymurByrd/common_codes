cmake_minimum_required(VERSION 3.4)
project(video_search)

set(CMAKE_CXX_STANDARD 11)

include(FeatureSummary)


set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")

set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})
set(SHARED_LIBRARY_VERSION ${VERSION_MAJOR}.${VERSION_MINOR})


if (CMAKE_INSTALL_PREFIX MATCHES "^/usr/local")
#if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) //only first time work
	set (CMAKE_INSTALL_PREFIX ../../deps)
endif()

# 默认选项配置
option(WITH_DE265  "Compile with libde265 decoder. (it is faster at x86 platform)" ON)
option(WITH_X265   "Compile with x265 encoder" ON)
option(WITH_A265   "Compile with a265 encoder" ON)
option(WITH_ITTIAM_DECODER   "Compile with ittiam decoder" ON)
#if (WITH_A265)
#	option(IPP_SUPPORT   "Compile ipp support" ON)
#else()
#	option(IPP_SUPPORT   "Compile ipp support" OFF)
#endif()
option(FPGA_SUPPORT  "Compile fpga support" OFF)

# 初始化参数为off
#add_feature_info(WITH_DE265          WITH_DE265    	"Compile with libde265 decoder. (it is faster at x86 platform)")
#add_feature_info(WITH_X265           WITH_X265     	"Compile with x265 encoder")
#add_feature_info(WITH_A265           WITH_A265     	"Compile with a265 encoder")
##add_feature_info(IPP_SUPPORT         OFF   	"Compile with a265 encoder. （must set WITH_A265 on）")
#add_feature_info(FPGA_SUPPORT        FPGA_SUPPORT  	"Fpga Hardware support")
#add_feature_info(WITH_ITTIAM_DECODER WITH_ITTIAM_DECODER    "Compile with ittiam decoder")
#
#if(NOT CMAKE_BUILD_TYPE)
#	set(CMAKE_BUILD_TYPE "Release" CACHE STRING
#			"Choose the type of build, standard options are: Debug Release RelWithDebInfo MinSizeRel."
#			FORCE)
#	add_feature_info(CMAKE_BUILD_TYPE 1 "Build type: ${CMAKE_BUILD_TYPE} (default)")
#else()
#	add_feature_info(CMAKE_BUILD_TYPE 1 "Build type: ${CMAKE_BUILD_TYPE} (selected)")
#endif()


execute_process(COMMAND hostname OUTPUT_VARIABLE HOSTNAME)
string(REPLACE "\n" "" HOSTNAME "${HOSTNAME}")
string(REPLACE "\r" "" HOSTNAME "${HOSTNAME}")
SET(COMMON_FLAGS "${COMMON_FLAGS} -DHOSTNAME=${HOSTNAME}")

execute_process(COMMAND git rev-parse HEAD OUTPUT_VARIABLE BUILD_COMMIT)
string(REPLACE "\n" "" BUILD_COMMIT "${BUILD_COMMIT}")
string(REPLACE "\r" "" BUILD_COMMIT "${BUILD_COMMIT}")
SET(COMMON_FLAGS "${COMMON_FLAGS} -DBUILD_COMMIT=${BUILD_COMMIT}")

MESSAGE(STATUS "BUILD_COMMIT: ${BUILD_COMMIT}")

#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static -ldl -lpthread")
#execute_process(COMMAND sh init_code.sh)

IF (CMAKE_C_COMPILER MATCHES "mclang$")
	MESSAGE(STATUS "mclang build ${CMAKE_C_COMPILER} For test only!!!!!")
	SET(FUZZ_FLAGS "-g -O2 -Wunused-const-variable -Wno-unused-variable -fno-omit-frame-pointer -fuzzer -afs-module=file-io -main-path=xheiftest_static. -fsanitize=address  -fprofile-instr-generate -fcoverage-mapping ")
#	SET(FUZZ_FLAGS "-g -O2 -Wunused-const-variable -Wno-unused-variable -fno-omit-frame-pointer -afs-module=file-io -main-path=fuzz_image. -fsanitize=address ")
	#export CC=/home/builder/Tools/build_toolchain/bin/mclang
	#export CXX="/home/builder/Tools/build_toolchain/bin/mclang++ -std=c++98"
ENDIF ()


IF (WIN32)
	MESSAGE(STATUS "Now is windows")
ELSEIF (APPLE)
	MESSAGE(STATUS "Now is Apple systens.")

	SET(IGNORE_WARNINGS "-Wno-unknown-pragmas -Wno-unused-command-line-argument -Wno-reorder -Wno-macro-redefined -Wno-unused-variable -Wno-comment -Wno-deprecated-declarations -Wno-unused-function -Wno-constant-conversion -Wno-incompatible-pointer-types")

	IF (NOT (CMAKE_BUILD_TYPE STREQUAL Debug))
		SET(COMMON_FLAGS 	"${COMMON_FLAGS} -march=native -Wall -pipe ${IGNORE_WARNINGS} -msse4.2 -mno-avx -DDEFAULT_ARCH=D_ARCH_X86_GENERIC")
	ELSE  (CMAKE_BUILD_TYPE STREQUAL Debug)
		SET(COMMON_FLAGS 	"${COMMON_FLAGS} -march=native -Wall -pipe ${IGNORE_WARNINGS} -msse4.2 -mno-avx -DDEFAULT_ARCH=D_ARCH_X86_GENERIC -fsanitize=address  -fprofile-instr-generate -fcoverage-mapping")
	ENDIF()

ELSEIF (UNIX)
	MESSAGE(STATUS "Now is UNIX-like OS's. ")

	SET(IGNORE_WARNINGS "-Wno-unknown-pragmas -Wno-comment -Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-function")
	SET(COMMON_FLAGS 	"${COMMON_FLAGS} -march=native -Wall -pipe ${IGNORE_WARNINGS} -msse4.2 -mno-avx -DDEFAULT_ARCH=D_ARCH_X86_GENERIC -fPIC ")

	set(MY_C_FLAGS   " -lstdc++ ")
	set(MY_CPP_FLAGS " -Wno-reorder ")
	set(MY_CXX_FLAGS " -Wno-reorder ")



ENDIF ()


set(CMAKE_C_FLAGS   "${FUZZ_FLAGS} ${COMMON_FLAGS} ${MY_C_FLAGS}")
set(CMAKE_CPP_FLAGS "${FUZZ_FLAGS} -lstdc++ ${COMMON_FLAGS} ${MY_CPP_FLAGS}")
set(CMAKE_CXX_FLAGS "${FUZZ_FLAGS} -lstdc++ ${COMMON_FLAGS} ${MY_CXX_FLAGS}")


file(GLOB test_src
		"video_search/video_separate.cpp"
		"video_search/video_search_orb.cpp"
		"video_search/video_search_hist.cpp"
		"main.cpp"
		)

include_directories(.)
include_directories(video_search)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)
link_directories(/usr/lib64/)


add_executable(video_search.bin ${test_src} )

if (HOSTNAME MATCHES "hujiapengdeMacBook-Pro.local")
	MESSAGE(STATUS "hostname: ${HOSTNAME}")
	find_package(OpenCV PATHS /usr/local/Cellar/opencv/4.1.2)
else ()
	MESSAGE(STATUS "hostname: ${HOSTNAME}")
	find_package(OpenCV)
ENDIF()

if (OpenCV_FOUND)
	MESSAGE(STATUS "OpenCV FOUND INC:${OpenCV_INCLUDE_DIRS} LIBS:${OpenCV_LIBRARY_DIRS}, ${OpenCV_LIBS}")
	include_directories(${OpenCV_INCLUDE_DIRS})
	link_directories(${OpenCV_LIBRARY_DIRS})
#	add_definitions(-DCV_TEST)

#	set(DEPS_LIBS ${DEPS_LIBS}   -lopencv_videoio -lopencv_imgproc -lopencv_core )
	set(DEPS_LIBS ${DEPS_LIBS}  ${OpenCV_LIBS} )
endif (OpenCV_FOUND)


find_package(BZip2)
if (BZIP2_FOUND)
	MESSAGE(STATUS "BZip2 FOUND INC:${BZIP2_INCLUDE_DIR} LIBS:${BZIP2_LIBRARIES}")
	include_directories(${BZIP2_INCLUDE_DIR})
	set(DEPS_LIBS ${DEPS_LIBS} ${BZIP2_LIBRARIES})
endif (BZIP2_FOUND)



IF (WIN32)
	MESSAGE(STATUS "Now is windows")
ELSEIF (APPLE)
	add_definitions(-DX86)
	add_definitions(-DDISABLE_AVX2)
	add_definitions(-DX64)
#	add_definitions(-DCV_TEST)

	add_definitions(-DAPPLE_ELF)
	add_definitions(-DHAVE_SSE)

ELSEIF (UNIX)
	link_directories(../../deps/lib64/)
	add_definitions(-DX86)
	add_definitions(-DDISABLE_AVX2)
	add_definitions(-DX64)

	set(DEPS_LIBS ${DEPS_LIBS} pthread rt)

	add_definitions(-DLINUX)
	add_definitions(-DHAVE_SSE)
ENDIF ()

set(DEPS_LIBS ${DEPS_LIBS} ${CMAKE_DL_LIBS} z )

target_link_libraries(video_search.bin ${DEPS_LIBS} )

FEATURE_SUMMARY(WHAT ALL INCLUDE_QUIET_PACKAGES)

MESSAGE(STATUS
	"The following libs to be linked:\n\n * ${DEPS_LIBS}\n")


#
#include_directories ("${PROJECT_SOURCE_DIR}/LibA265Encoder/interface/")
#add_subdirectory (LibA265Encoder)
#
#include_directories ("${PROJECT_SOURCE_DIR}/LibSccEncoder/interface/")
#add_subdirectory (LibSccEncoder)



