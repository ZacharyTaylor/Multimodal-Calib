include_directories ("${PROJECT_SOURCE_DIR}/Code/CI")
include_directories ("${PROJECT_SOURCE_DIR}/Code/CI/code")
include_directories ("${PROJECT_SOURCE_DIR}/Code/CI/code/internal")

FILE(GLOB HEADERS *.h)
FILE(GLOB SOURCE *.cu)

cuda_add_library (Multimodal-Calib SHARED ${SOURCE} ${HEADERS})
