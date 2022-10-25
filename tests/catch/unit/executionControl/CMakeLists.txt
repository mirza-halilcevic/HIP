set(TEST_SRC
    hipFuncSetCacheConfig.cc
    hipFuncSetSharedMemConfig.cc
    hipFuncSetAttribute.cc
)

hip_add_exe_to_target(NAME ExecutionControlTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)