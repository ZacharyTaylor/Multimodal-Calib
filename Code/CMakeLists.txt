include_directories ("${PROJECT_SOURCE_DIR}/Code/CI")
include_directories ("${PROJECT_SOURCE_DIR}/Code/Reduce")

FILE(GLOB HEADERS *.h)
FILE(GLOB SOURCE *.cu)

cuda_add_library (MultiCalLib ${SOURCE} ${HEADERS})