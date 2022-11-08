set(TEST_SRC
    hipGLGetDevices.cc
)

hip_add_exe_to_target(NAME InteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
target_link_libraries(InteropTest OpenGL::GL OpenGL::EGL)