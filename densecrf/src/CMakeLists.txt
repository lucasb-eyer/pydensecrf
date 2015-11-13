include_directories( ../external/liblbfgs/include ../external/ )

add_library( densecrf util.cpp permutohedral.cpp unary.cpp pairwise.cpp objective.cpp densecrf.cpp labelcompatibility.cpp )

add_library( optimization optimization.cpp )
target_link_libraries( optimization lbfgs )
