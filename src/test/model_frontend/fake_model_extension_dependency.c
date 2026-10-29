#if defined(_WIN32)
#define FLM_TEST_EXPORT __declspec(dllexport)
#else
#define FLM_TEST_EXPORT __attribute__((visibility("default")))
#endif

/* Indexed by the fake extension's FLM_TEST_EVENT_* values. Kept here because
 * the host test links this library too, so both sides see one counter set. */
static int g_event_counts[2];

FLM_TEST_EXPORT int flm_test_model_extension_dependency(void) {
    return 1;
}

FLM_TEST_EXPORT void flm_test_record_event(int event) {
    if (event >= 0 && event < (int)(sizeof(g_event_counts) /
                                    sizeof(g_event_counts[0]))) {
        ++g_event_counts[event];
    }
}

FLM_TEST_EXPORT int flm_test_event_count(int event) {
    if (event < 0 || event >= (int)(sizeof(g_event_counts) /
                                    sizeof(g_event_counts[0]))) {
        return -1;
    }
    return g_event_counts[event];
}
