#if defined(_WIN32)
#define FLM_TEST_EXPORT __declspec(dllexport)
#else
#define FLM_TEST_EXPORT __attribute__((visibility("default")))
#endif

FLM_TEST_EXPORT int flm_test_model_extension_dependency(void) {
    return 1;
}
