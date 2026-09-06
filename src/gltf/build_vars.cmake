# add draco support if available
find_package(draco)

if(draco_FOUND)
    OPTION(vsgXchange_draco "Optional glTF draco support provided" ON)
endif()

# add EXT_meshopt_compression support if available
find_package(meshoptimizer CONFIG)

if(meshoptimizer_FOUND)
    OPTION(vsgXchange_meshoptimizer "Optional glTF EXT_meshopt_compression support provided" ON)
endif()

set(SOURCES ${SOURCES}
    gltf/gltf.cpp
    gltf/SceneGraphBuilder.cpp
)

if (draco_FOUND AND vsgXchange_draco)
    set(EXTRA_INCLUDES ${EXTRA_INCLUDES} ${draco_INCLUDE_DIRS})
    set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} draco::draco)
    if(NOT BUILD_SHARED_LIBS)
        set(FIND_DEPENDENCY ${FIND_DEPENDENCY} "find_dependency(draco)")
    endif()
endif()

if (meshoptimizer_FOUND AND vsgXchange_meshoptimizer)
    set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} meshoptimizer::meshoptimizer)
    if(NOT BUILD_SHARED_LIBS)
        set(FIND_DEPENDENCY ${FIND_DEPENDENCY} "find_dependency(meshoptimizer CONFIG)")
    endif()
endif()
