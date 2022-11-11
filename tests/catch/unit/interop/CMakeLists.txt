set(TEST_SRC
    vulkan_test.cc
    hipImportExternalSemaphore.cc
    hipDestroyExternalSemaphore.cc
    hipWaitExternalSemaphoresAsync.cc
    hipSignalExternalSemaphoresAsync.cc
   )

hip_add_exe_to_target(NAME InteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS vulkan)