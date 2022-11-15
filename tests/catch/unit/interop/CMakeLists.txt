find_package(hip REQUIRED)

set(TEST_SRC
    vulkan_test.cc
    hipImportExternalSemaphore.cc
    hipWaitExternalSemaphoresAsync.cc
    hipSignalExternalSemaphoresAsync.cc
    hipDestroyExternalSemaphore.cc
    hipImportExternalMemory.cc
    hipExternalMemoryGetMappedBuffer.cc
    hipDestroyExternalMemory.cc
   )

hip_add_exe_to_target(NAME InteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS vulkan)