file(GLOB CNNV_TEST_SRC CONFIGURE_DEPENDS *.cpp)

add_executable(cnnv_tests ${CNNV_TEST_SRC})
target_include_directories(cnnv_tests PRIVATE .)
target_compile_options(cnnv_tests PRIVATE ${CNNV_WARNING_FLAGS})
target_link_libraries(cnnv_tests PRIVATE cnnv_core)

add_test(NAME cnnv_tests COMMAND cnnv_tests)
