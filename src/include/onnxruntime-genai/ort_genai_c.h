// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef BUILDING_ORT_GENAI_C
#define OGA_EXPORT __declspec(dllexport)
#else
#define OGA_EXPORT __declspec(dllimport)
#endif
#define OGA_API_CALL _stdcall
#else
// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define OGA_EXPORT __attribute__((visibility("default")))
#else
#define OGA_EXPORT
#endif
#define OGA_API_CALL
#endif

/** \addtogroup Global
 * ONNX Runtime Generative AI C API
 * This API is not thread safe.
 * @{
 */

typedef enum OgaElementType {
  OgaElementType_undefined,
  OgaElementType_float32,     // maps to c type float
  OgaElementType_uint8,       // maps to c type uint8_t
  OgaElementType_int8,        // maps to c type int8_t
  OgaElementType_uint16,      // maps to c type uint16_t
  OgaElementType_int16,       // maps to c type int16_t
  OgaElementType_int32,       // maps to c type int32_t
  OgaElementType_int64,       // maps to c type int64_t
  OgaElementType_string,      // string type (not currently supported by Oga)
  OgaElementType_bool,        // maps to c type bool
  OgaElementType_float16,     // IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
  OgaElementType_float64,     // maps to c type double
  OgaElementType_uint32,      // maps to c type uint32_t
  OgaElementType_uint64,      // maps to c type uint64_t
  OgaElementType_complex64,   // complex with float32 real and imaginary components
  OgaElementType_complex128,  // complex with float64 real and imaginary components
  OgaElementType_bfloat16,    // Non-IEEE floating-point format based on IEEE754 single-precision
} OgaElementType;

typedef struct OgaResult OgaResult;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaSpeculativeStats OgaSpeculativeStats;
typedef struct OgaRuntimeSettings OgaRuntimeSettings;
typedef struct OgaConfig OgaConfig;
typedef struct OgaModel OgaModel;
// OgaSequences is an array of token arrays where the number of token arrays can be obtained using
// OgaSequencesCount and the number of tokens in each token array can be obtained using OgaSequencesGetSequenceCount.
typedef struct OgaSequences OgaSequences;
typedef struct OgaTokenizer OgaTokenizer;
typedef struct OgaTokenizerStream OgaTokenizerStream;
typedef struct OgaTensor OgaTensor;
typedef struct OgaImages OgaImages;
typedef struct OgaNamedTensors OgaNamedTensors;
typedef struct OgaMultiModalProcessor OgaMultiModalProcessor;
typedef struct OgaAudios OgaAudios;
typedef struct OgaStringArray OgaStringArray;
typedef struct OgaAdapters OgaAdapters;
typedef struct OgaEngine OgaEngine;
typedef struct OgaEngineCapabilities OgaEngineCapabilities;
typedef struct OgaEngineEvent OgaEngineEvent;
typedef struct OgaEngineEventBuffer OgaEngineEventBuffer;
typedef struct OgaRequest OgaRequest;
typedef struct OgaRequestOptions OgaRequestOptions;
typedef struct OgaTurnOptions OgaTurnOptions;
typedef struct OgaTurnUsage OgaTurnUsage;
typedef struct OgaStreamingProcessor OgaStreamingProcessor;

/**
 * \brief Reason why an Engine Request's generation turn stopped.
 */
typedef uint32_t OgaFinishReason;
#define OgaFinishReason_None ((OgaFinishReason)0)
#define OgaFinishReason_Eos ((OgaFinishReason)1)
#define OgaFinishReason_StopString ((OgaFinishReason)2)
#define OgaFinishReason_MaxGeneratedTokens ((OgaFinishReason)3)
#define OgaFinishReason_MaxSessionTokens ((OgaFinishReason)4)
#define OgaFinishReason_Cancelled ((OgaFinishReason)5)
#define OgaFinishReason_Failed ((OgaFinishReason)6)

/** \brief Bit flags describing the payload present in an OgaEngineEvent. */
typedef uint32_t OgaEngineEventFlags;
#define OgaEngineEventFlag_None ((OgaEngineEventFlags)0)
#define OgaEngineEventFlag_Token ((OgaEngineEventFlags)(1u << 0))
#define OgaEngineEventFlag_TurnFinished ((OgaEngineEventFlags)(1u << 1))
#define OgaEngineEventFlag_CapacityBlocked ((OgaEngineEventFlags)(1u << 2))
#define OgaEngineEventFlag_Failed ((OgaEngineEventFlags)(1u << 3))
#define OgaEngineEventFlag_Retryable ((OgaEngineEventFlags)(1u << 4))

/** \brief Stable behavioral classification for an Engine event. */
typedef uint32_t OgaErrorCode;
#define OgaErrorCode_None ((OgaErrorCode)0)
#define OgaErrorCode_CapacityDeferred ((OgaErrorCode)1)
#define OgaErrorCode_ExecutionCapacityExceeded ((OgaErrorCode)2)
#define OgaErrorCode_RetryableExecution ((OgaErrorCode)3)
#define OgaErrorCode_RequestUnserviceable ((OgaErrorCode)4)
#define OgaErrorCode_EngineContractFailure ((OgaErrorCode)5)
#define OgaErrorCode_EngineExecutionFailure ((OgaErrorCode)6)

//! @}

/** \addtogroup Global
 * @{
 */

/**
 * \brief Shuts down the GenAI library and releases all GenAI-owned ONNX Runtime globals.
 *
 * \warning Callers SHOULD invoke OgaShutdown() before process exit. If it is not called, GenAI's globals are destroyed
 *          at static-destruction time in undefined order, which may crash.
 *
 * \note C++ callers should prefer the OgaHandle RAII wrapper in ort_genai.h; C# callers should prefer the
 *       OgaHandle IDisposable wrapper. Both invoke OgaShutdown() on destruction.
 *
 * \note Lifetime contract: no OgaModel / OgaGenerator / OgaTokenizer / OgaTensor / OgaEngine /
 *       OgaEngineEventBuffer / OgaRequest, or any object that holds device memory, may outlive
 *       OgaShutdown(). The caller MUST destroy every such object before calling OgaShutdown().
 *       Calling OgaShutdown() with such objects still alive is undefined behavior (typically a
 *       crash when the buffer is freed through a now-invalid allocator).
 *
 * \note Re-initialization: OgaShutdown() is a full teardown -- it destroys GenAI's ONNX Runtime environment and unloads
 *       GenAI's add-on libraries. GenAI may be used again after OgaShutdown(); the next GenAI call re-initializes with a
 *       fresh environment.
 *
 * \note If a host registered execution provider libraries directly on its own OrtEnv reference, it should unregister
 *       them (on that reference) and release the reference after OgaShutdown(), once all GenAI usage is finished.
 */
OGA_EXPORT void OGA_API_CALL OgaShutdown();

/**
 * \brief Enable or disable non-essential telemetry event collection.
 *
 * Telemetry can be fully disabled at compile time (ENABLE_TELEMETRY=OFF) or by setting
 * ORT_DISABLE_TELEMETRY=1 before initialization. The environment variable prevents the uploader,
 * events, and persistent device identifier from being created for the process lifetime. Runtime
 * controls suppress new detailed model, generation, adapter, and error telemetry; the process
 * information event may still be emitted.
 *
 * \param[in] enabled true to enable telemetry, false to disable.
 */
OGA_EXPORT void OGA_API_CALL OgaSetTelemetryEnabled(bool enabled);

/**
 * \param[in] result OgaResult that contains the error message.
 * \return Error message contained in the OgaResult. The const char* is owned by the OgaResult
 *         and can will be freed when the OgaResult is destroyed.
 */
OGA_EXPORT const char* OGA_API_CALL OgaResultGetError(const OgaResult* result);

/**
 * \brief Control the logging behavior of the library.
 *        If OgaSetLogString is called with name "filename", and value is a valid file path,
 *        the library will log to that file. This will override any previously set logging destination.
 *        If OgaSetLogString is called with name "filename" and the value provided is an empty string,
 *        the library will log to the default destination (i.e. std::cerr) thereafter.
 * \param[in] name logging option name, see logging.h 'struct LogItems' for the list of available options
 * \param[in] value logging option value.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetLogBool(const char* name, bool value);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetLogString(const char* name, const char* value);

/**
 * \brief Register a callback function to receive log messages from the library. If invoked, the callback will override
 *        the previously set logging destination (e.g. a file or std::cerr).
 * \param[in] callback function pointer to the logging callback function (use nullptr to disable callback and revert to
 *                     the default logging destination - std::cerr).
 * \return OgaResult containing the error message when the callback could not be set, else nullptr.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetLogCallback(void (*callback)(const char* string, size_t length));

/**
 * \param[in] result OgaResult to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyResult(OgaResult* result);
OGA_EXPORT void OGA_API_CALL OgaDestroyString(const char*);
OGA_EXPORT void OGA_API_CALL OgaDestroyNamedTensors(OgaNamedTensors*);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateSequences(OgaSequences** out);

/**
 * \param[in] sequences OgaSequences to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroySequences(OgaSequences* sequences);

/**
 * \brief Returns the number of sequences in the OgaSequences
 * \param[in] sequences
 * \return The number of sequences in the OgaSequences
 */
OGA_EXPORT size_t OGA_API_CALL OgaSequencesCount(const OgaSequences* sequences);

/**
 * \brief Appends token_cnt number of tokens from token_ptr to sequence
 * \param[in] token_ptr constant pointer to int32 tokens
 * \param[in] token_cnt number of tokens to read from token_ptr
 * \param[in] sequences OgaSequences object to append the tokens to
 * \return OgaResult containing the error message when tokens could not been added, else nullptr.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaAppendTokenSequence(const int32_t* token_ptr, size_t token_cnt, OgaSequences* sequences);

/**
 * \brief Appends the given token to the sequence at the given index.
          If the sequence at the given index does not exist, a new sequence is
          created at the given index if sequence_idx is equal to the current sequences count.
 * \param[in] token token to append to the sequence
 * \param[in] sequences OgaSequences object to append the token to
 * \param[in] sequence_index index of the sequence to append the token to
 * \return OgaResult containing the error message when tokens could not been added, else nullptr.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaAppendTokenToSequence(int32_t token, OgaSequences* sequences, size_t sequence_index);

/**
 * \brief Returns the number of tokens in the sequence at the given index.
 * \param[in] sequences OgaSequences to use.
 * \param[in] sequence_index index of the sequence to use.
 * \return The number of tokens in the sequence at the given index. Returns 0 if
 *         sequence_index is out of bounds (i.e. >= OgaSequencesCount(sequences)).
 */
OGA_EXPORT size_t OGA_API_CALL OgaSequencesGetSequenceCount(const OgaSequences* sequences, size_t sequence_index);

/**
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaSequencesGetSequenceCount
 * \param[in] sequences OgaSequences to use.
 * \param[in] sequence_index index of the sequence to use.
 * \return The pointer to the sequence data at the given index. The pointer is valid until the OgaSequences is destroyed.
 *         Returns nullptr if sequence_index is out of bounds (i.e. >= OgaSequencesCount(sequences)).
 */
OGA_EXPORT const int32_t* OGA_API_CALL OgaSequencesGetSequenceData(const OgaSequences* sequences, size_t sequence_index);

OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadImage(const char* image_path, OgaImages** images);
OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadImages(const OgaStringArray* image_paths, OgaImages** images);

/**
 * \brief Load multiple images from an array of byte buffers
 * \param[in] image_data Array of byte buffers containing the image data.
 * \param[in] image_data_sizes Array of sizes of the byte buffers.
 * \param[in] count Number of images to load.
 * \param[out] images The loaded images.
 * \return OgaResult containing the error message if the loading of the images failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadImagesFromBuffers(const void** image_data, const size_t* image_data_sizes, size_t count, OgaImages** images);

OGA_EXPORT void OGA_API_CALL OgaDestroyImages(OgaImages* images);

OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadAudio(const char* audio_path, OgaAudios** audios);

OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadAudios(const OgaStringArray* audio_paths, OgaAudios** audios);

/**
 * \brief Load multiple audios from an array of byte buffers
 * \param[in] audio_data Array of byte buffers containing the audio data.
 * \param[in] audio_data_sizes Array of sizes of the byte buffers.
 * \param[in] count Number of audios to load.
 * \param[out] audios The loaded audios.
 * \return OgaResult containing the error message if the loading of the audios failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadAudiosFromBuffers(const void** audio_data, const size_t* audio_data_sizes, size_t count, OgaAudios** audios);

OGA_EXPORT void OGA_API_CALL OgaDestroyAudios(OgaAudios* audios);

/**
 * \brief Creates a runtime settings instance to be used to create a model.
 * \param[out] out The created runtime settings.
 * \return OgaResult containing the error message if the creation of the runtime settings failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateRuntimeSettings(OgaRuntimeSettings** out);
/**
 * \brief Destroys the given runtime settings.
 * \param[in] settings The runtime settings to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyRuntimeSettings(OgaRuntimeSettings* settings);

/**
 * \brief Sets a specific runtime handle for the runtime settings.
 * \param[in] settings The runtime settings to set the device type.
 * \param[in] handle_name The name of the handle to set for the runtime settings.
 * \param[in] handle The value of handle to set for the runtime settings.
 * \return OgaResult containing the error message if the setting of the device type failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRuntimeSettingsSetHandle(OgaRuntimeSettings* settings, const char* handle_name, void* handle);

/**
 * \brief Creates an OgaConfig from the given configuration directory.
 * \param[in] config_path The path to the configuration directory. The path is expected to be encoded in UTF-8.
 * \param[out] out The created config.
 * \return OgaResult containing the error message if the creation of the config failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateConfig(const char* config_path, OgaConfig** out);

/**
 * \brief Creates an OgaConfig from a model package directory, using the supplied execution
 *        provider to select a variant.
 *
 * `config_path` must refer to a model package; passing a flat directory returns an error.
 * When the package declares exactly one execution provider across its variants, `ep` may
 * be null or empty (the EP is auto-detected). Otherwise `ep` selects which variant is
 * loaded.
 *
 * To load a model with an explicit EP, pass the resulting OgaConfig to
 * OgaCreateModelFromConfig. OgaCreateModel does not accept an ep argument because the
 * config carries everything needed.
 *
 * \param[in] config_path Path to the model package, encoded in UTF-8.
 * \param[in] ep Execution provider name, or null/empty for auto-detection.
 * \param[out] out The created config.
 * \return OgaResult containing the error message on failure.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateConfigFromPackageEp(const char* config_path, const char* ep, OgaConfig** out);

/**
 * \brief Clear the list of providers in the given config
 * \param[in] config The config to clear the providers from.
 * \return OgaResult containing the error message if the clearing of the providers failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigClearProviders(OgaConfig* config);

/**
 * \brief Add the provider at the end of the list of providers in the given config if it doesn't already exist.
 * If it already exists, do nothing.
 * \param[in] config The config to set the provider on.
 * \param[in] provider The provider to set on the config.
 * \return OgaResult containing the error message if the setting of the provider failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigAppendProvider(OgaConfig* config, const char* provider);

/**
 * \brief Set a provider option
 * \param[in] config The config to set the provider option on.
 * \param[in] provider The provider to set the option on.
 * \param[in] key The key of the option to set.
 * \param[in] value The value of the option to set.
 * \return OgaResult containing the error message if the setting of the provider option failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigSetProviderOption(OgaConfig* config, const char* provider, const char* key, const char* value);

/**
 * \brief Add the model data to load the model from memory. Applications may call OgaConfigRemoveModelData to remove the model data
 *        when it is no longer needed.
 *
 * Note that the model data is expected to be valid at least until the model is created.
 * If using session options such as `session.use_ort_model_bytes_directly`, the model data must remain valid
 * until the OgaModel is destroyed, as the model data will be used directly by the Ort::Session.
 * Please see the relevant ONNX Runtime documentation for more details on this option.
 *
 * \param[in] config The config to add the model data to.
 * \param[in] model_filename The name of the model file as defined in the config.
 * \param[in] model_data The model data to add. The data is expected to be valid at least until the model is created.
 * \param[in] model_data_length The length of the model data.
 * \return OgaResult containing the error message if the addition of the model data failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigAddModelData(OgaConfig* config, const char* model_filename, const void* model_data, size_t model_data_length);

/**
 * \brief Remove model data previously added to the config.
 * \param[in] config The config to remove the model data from.
 * \param[in] model_filename The name of the model file as defined in the config.
 * \return OgaResult containing the error message if the removal of the model data failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigRemoveModelData(OgaConfig* config, const char* model_filename);

/**
 * \brief Filter EP devices by hardware device type property with ONNXRuntime API.
 * \param[in] config The config to overlay the JSON on.
 * \param[in] hardware_device_type hardware device type, e.g., CPU, GPU, NPU.
 * \return OgaResult containing the error message if the overlaying of the JSON failed.
 *
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigSetDecoderProviderOptionsHardwareDeviceType(OgaConfig* config, const char* provider, const char* hardware_device_type);

/**
 * \brief Filter EP devices by hardware device id property with ONNXRuntime API.
 * \param[in] config The config to overlay the JSON on.
 * \param[in] hardware_device_type hardware device id.
 * \return OgaResult containing the error message if the overlaying of the JSON failed.
 *
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigSetDecoderProviderOptionsHardwareDeviceId(OgaConfig* config, const char* provider, uint32_t hardware_device_id);

/**
 * \brief Filter EP devices by hardware vendor id property with ONNXRuntime API.
 * \param[in] config The config to overlay the JSON on.
 * \param[in] hardware_device_type hardware vendor id.
 * \return OgaResult containing the error message if the overlaying of the JSON failed.
 *
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigSetDecoderProviderOptionsHardwareVendorId(OgaConfig* config, const char* provider, uint32_t hardware_vendor_id);

/**
 * \brief Clear the hardware device type property
 * \param[in] config The config to clear hardware device type property.
 * \return OgaResult containing the error message if the clearing failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigClearDecoderProviderOptionsHardwareDeviceType(OgaConfig* config, const char* provider);

/**
 * \brief Clear the hardware device id property
 * \param[in] config The config to clear hardware device id property.
 * \return OgaResult containing the error message if the clearing failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigClearDecoderProviderOptionsHardwareDeviceId(OgaConfig* config, const char* provider);

/**
 * \brief Clear the hardware vendor id property
 * \param[in] config The config to clear hardware vendor id property.
 * \return OgaResult containing the error message if the clearing failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigClearDecoderProviderOptionsHardwareVendorId(OgaConfig* config, const char* provider);

/**
 * \brief Overlay JSON on top of config file
 * \param[in] config The config to overlay the JSON on.
 * \param[in] json The JSON to overlay on the config.
 * \return OgaResult containing the error message if the overlaying of the JSON failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaConfigOverlay(OgaConfig* config, const char* json);

/**
 * \brief Creates a model from the given configuration directory.
 * \param[in] config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * \param[out] out The created model.
 * \return OgaResult containing the error message if the model creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out);

/**
 * \brief Creates a model from the given configuration.
 * \param[in] config The configuration to use for the model.
 * \param[out] out The created model.
 * \return OgaResult containing the error message if the model creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModelFromConfig(const OgaConfig* config, OgaModel** out);

/**
 * \brief Creates a model from the given configuration directory, runtime settings and device type.
 * \param[in] config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * \param[in] settings The runtime settings to use for the model.
 * \param[out] out The created model.
 * \return OgaResult containing the error message if the model creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModelWithRuntimeSettings(const char* config_path, const OgaRuntimeSettings* settings, OgaModel** out);

/**
 * \brief Returns the type of the model.
 * \param[in] model The model to get the type from.
 * \param[out] out The type of the model. Must be destroyed with OgaDestroyString
 * \return OgaResult containing the error message if the getting of the model type failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaModelGetType(const OgaModel* model, const char** out);

/**
 * \brief Returns the device type of the model.
 * \param[in] model The model to get the device type from.
 * \param[out] out The device type of the model. Must be destroyed with OgaDestroyString
 * \return OgaResult containing the error message if the getting of the device type failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaModelGetDeviceType(const OgaModel* model, const char** out);

/**
 * \brief Destroys the given config
 * \param[in] config The config to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyConfig(OgaConfig* config);

/**
 * \brief Destroys the given model.
 * \param[in] model The model to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyModel(OgaModel* model);

/**
 * \brief Creates a OgaGeneratorParams from the given model.
 * \param[in] model The model to use for generation.
 * \param[out] out The created generator params.
 * \return OgaResult containing the error message if the generator params creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out);

/**
 * \brief Destroys the given generator params.
 * \param[in] params The generator params to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* params);

/**
 * \brief Set a numerical value for a search parameter
 * \param[in] params The generator params to set.
 * \param[in] name The name of the search parameter.
 * \param[in] value The value of the search parameter.
 * \return OgaResult containing the error message if setting the generator params failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchNumber(OgaGeneratorParams* params, const char* name, double value);

/**
 * \brief Set a boolean value for a search parameter
 * \param[in] params The generator params to set.
 * \param[in] name The name of the search parameter.
 * \param[in] value The value of the search parameter.
 * \return OgaResult containing the error message if setting the generator params failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchBool(OgaGeneratorParams* params, const char* name, bool value);

/**
 * \brief Sets the guidance type and data for the Generator params
 * \param[in] params The generator params to set the guidance on
 * \param[in] type The type of the guidance. Currently, we support json_schema, regex and lark_grammar.
 * \param[in] data The input string, which is the guidance data.
 * \param[in] enable_ff_tokens Whether to enable ff_tokens generation. This feature allows guidance to force-forward tokens that satisfy input grammar without calling model, hence speeding up generation process. Only valid when guidance type is set and batch_size is 1 and beam_size is 1.
 * \return OgaResult containing the error message if the setting of the guidance failed
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetGuidance(OgaGeneratorParams* params, const char* type, const char* data, bool enable_ff_tokens);

/**
 * \brief Get a numerical value for a search parameter
 * \param[in] params The generator params to set.
 * \param[in] name The name of the search parameter.
 * \param[out] value The value of the search parameter.
 * \return OgaResult containing the error message if setting the generator params failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsGetSearchNumber(const OgaGeneratorParams* params, const char* name, double* value);

/**
 * \brief Get a boolean value for a search parameter
 * \param[in] params The generator params to set.
 * \param[in] name The name of the search parameter.
 * \param[out] value The value of the search parameter.
 * \return OgaResult containing the error message if setting the generator params failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsGetSearchBool(const OgaGeneratorParams* params, const char* name, bool* value);

/**
 * \brief Set a numerical value for a speculative decoding option.
 * \param[in] params The generator params to set.
 * \param[in] name The name of the speculative option (e.g. "max_draft_tokens" or "ngram_size").
 * \param[in] value The value to set.
 * \return OgaResult containing the error message if setting the option failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSpeculativeNumber(OgaGeneratorParams* params, const char* name, double value);

/**
 * \brief Get a numerical value for a speculative decoding option.
 * \param[in] params The generator params to query.
 * \param[in] name The name of the speculative option.
 * \param[out] value The current value.
 * \return OgaResult containing the error message if getting the option failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsGetSpeculativeNumber(const OgaGeneratorParams* params, const char* name, double* value);

/**
 * \brief Set a boolean value for a speculative decoding option.
 * \param[in] params The generator params to set.
 * \param[in] name The name of the speculative option.
 * \param[in] value The value to set.
 * \return OgaResult containing the error message if setting the option failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSpeculativeBool(OgaGeneratorParams* params, const char* name, bool value);

/**
 * \brief Get a boolean value for a speculative decoding option.
 * \param[in] params The generator params to query.
 * \param[in] name The name of the speculative option.
 * \param[out] value The current value.
 * \return OgaResult containing the error message if getting the option failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsGetSpeculativeBool(const OgaGeneratorParams* params, const char* name, bool* value);

/**
 * \brief Creates a generator from the given model and generator params.
 * \param[in] model The model to use for generation.
 * \param[in] params The parameters to use for generation.
 * \param[out] out The created generator.
 * \return OgaResult containing the error message if the generator creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out);

/**
 * \brief Destroys the given generator.
 * \param[in] generator The generator to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyGenerator(OgaGenerator* generator);

/* Multi-Token-Prediction (MTP) self-speculative decoding (e.g. Qwen3.6). The MtpGenerator
 * composes a main decoder and an MTP draft head and runs the draft/verify loop in-engine,
 * keeping the hidden-state handoff on-device. Greedy, batch size 1; the output is identical to
 * plain greedy decoding (lossless). */
typedef struct OgaMtpGenerator OgaMtpGenerator;

/**
 * \brief Creates an MTP self-speculative generator from a main model and an MTP-head model.
 * \param[in] main_model The main decoder (exported with include_hidden_states).
 * \param[in] mtp_model The MTP head (mtp.onnx, with a hidden_states input).
 * \param[in] params The generation parameters (greedy, batch size 1).
 * \param[out] out The created MTP generator.
 * \return OgaResult containing the error message on failure.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateMtpGenerator(const OgaModel* main_model, const OgaModel* mtp_model, const OgaGeneratorParams* params, OgaMtpGenerator** out);

/** \brief Seed the prompt (runs the main model's prefill). */
OGA_EXPORT OgaResult* OGA_API_CALL OgaMtpGenerator_AppendTokens(OgaMtpGenerator* generator, const int32_t* input_ids, size_t input_ids_count);
/** \brief Produce the next token via the draft/verify loop. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaMtpGenerator_GenerateNextToken(OgaMtpGenerator* generator);
/** \brief Reset to an un-primed, empty request while retaining allocated state and captured graphs.
 * Call OgaMtpGenerator_AppendTokens before generating again. This invalidates any pointer
 * previously returned by OgaMtpGenerator_GetSequenceData.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaMtpGenerator_Reset(OgaMtpGenerator* generator);
/** \brief Whether generation has reached EOS or max length. */
OGA_EXPORT bool OGA_API_CALL OgaMtpGenerator_IsDone(const OgaMtpGenerator* generator);
/** \brief Number of committed tokens (prompt + generated). */
OGA_EXPORT size_t OGA_API_CALL OgaMtpGenerator_GetSequenceCount(const OgaMtpGenerator* generator);
/** \brief Pointer to the committed token data (valid until the next call). */
OGA_EXPORT const int32_t* OGA_API_CALL OgaMtpGenerator_GetSequenceData(const OgaMtpGenerator* generator);
/** \brief Number of main-model forward passes run (speculative stats). */
OGA_EXPORT size_t OGA_API_CALL OgaMtpGenerator_GetForwardCount(const OgaMtpGenerator* generator);
/** \brief Number of accepted drafts (speculative stats). */
OGA_EXPORT size_t OGA_API_CALL OgaMtpGenerator_GetAcceptCount(const OgaMtpGenerator* generator);
/** \brief Number of draft attempts (speculative stats). */
OGA_EXPORT size_t OGA_API_CALL OgaMtpGenerator_GetTrialCount(const OgaMtpGenerator* generator);
/** \brief Returns speculative-decoding statistics using the shared OgaSpeculativeStats schema. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaMtpGenerator_GetSpeculativeStats(const OgaMtpGenerator* generator, OgaSpeculativeStats** out);
/** \brief Destroys the given MTP generator. */
OGA_EXPORT void OGA_API_CALL OgaDestroyMtpGenerator(OgaMtpGenerator* generator);

/**
 * \brief Returns true if the generator has finished generating all the sequences.
 * \param[in] generator The generator to check if it is done with generating all sequences.
 * \return True if the generator has finished generating all the sequences, false otherwise.
 */
OGA_EXPORT bool OGA_API_CALL OgaGenerator_IsDone(OgaGenerator* generator);

/**
 * \brief Returns true if the session has been terminated.
 * \param[in] generator The generator to add the inputs to.
 * \return True if the session has been terminated, false otherwise.
 */
OGA_EXPORT bool OGA_API_CALL OgaGenerator_IsSessionTerminated(const OgaGenerator* generator);

/**
 * \brief For additional model inputs that genai does not handle, this lets the user set their values. For example LoRA models handle
 * fine tuning through model inputs. This lets the user supply the fine tuning inputs, while genai handles the standard inputs.
 * \param[in] generator The generator to add the inputs to.
 * \param[in] name Name of the model input (this must match the model's input name)
 * \param[in] tensor The OgaTensor of the input data
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_SetModelInput(OgaGenerator* generator, const char* name, OgaTensor* tensor);

/**
 * \brief For additional model inputs that genai does not handle, this lets the user set their values.
 * \param[in] generator The generator to add the inputs to.
 * \param[in] named_tensors The named tensors to set the inputs as.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_SetInputs(OgaGenerator* generator, const OgaNamedTensors* named_tensors);

/**
 * \brief Adds the input ids to the generator. The input ids are used to seed the generation.
 * \param[in] generator The generator to add the input ids to.
 * \param[in] p_sequences The input id sequences.
 * \return OgaResult containing the error message if the setting of the input ids failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_AppendTokenSequences(OgaGenerator* generator, const OgaSequences* p_sequences);

/**
 * \brief Adds the input ids to the generator. The input ids are used to seed the generation.
 * \param[in] generator The generator to add the input ids to.
 * \param[in] input_ids The input ids to add.
 * \param[in] input_ids_count The number of input ids to add (batch_size * sequence_length).
 * \return OgaResult containing the error message if the setting of the input ids failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_AppendTokens(OgaGenerator* generator, const int32_t* input_ids, size_t input_ids_count);

/**
 * \brief Returns the number of tokens in the generator
 * \param[in] generator The generator containing the appended tokens.
 * \return The number of tokens that have been added.
 */
OGA_EXPORT size_t OGA_API_CALL OgaGenerator_TokenCount(const OgaGenerator* generator);

/**
 * \brief Computes the logits from the model based on the input ids and the past state. The computed logits are stored in the generator.
 * \param[in] generator The generator to compute the logits for.
 * \return OgaResult containing the error message if the computation of the logits failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken(OgaGenerator* generator);

/**
 * \brief Returns a pointer to the next tokens generated by the model. The out_count will match the batch size
 * \param[in] generator The generator to get the next tokens from.
 * \param[out] out The pointer to the next tokens generated by the model. The pointer is valid until the next OgaGenerator call
 * \param[out] out_count The number of tokens in the out array.
 * \return OgaResult containing the error message if the getting of the next tokens failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GetNextTokens(const OgaGenerator* generator, const int32_t** out, size_t* out_count);

/**
 * \brief Set a runtime option's name and value.
 * \param[in] generator The generator to rewind to the given length.
 * \param[in] key The runtime option's name
 * \param[in] value The runtime option's value
 * \return OgaResult containing the error message if setting the runtime option failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_SetRuntimeOption(OgaGenerator* generator, const char* key, const char* value);

/**
 * \brief Rewinds the generator to the given length. This is useful when the user wants to rewind the generator to a specific length
 *        and continue generating from that point. Rewinding is not supported with beam search.
 * \param[in] generator The generator to rewind to the given length.
 * \param[in] new_length The desired length in tokens after rewinding.
 * \return OgaResult containing the error message if the rewinding failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_RewindTo(OgaGenerator* generator, size_t new_length);

/**
 * \brief Snapshots the generator's recurrent state (conv/linear-attention state) so a later
 *        OgaGenerator_RewindTo can roll it back. Used for speculative decoding (e.g. MTP) on
 *        hybrid models, whose recurrent state cannot be partially cropped like the KV cache.
 *        Take the snapshot at the length you might roll back to, then RewindTo that length to
 *        restore it. A no-op for models without recurrent state.
 * \param[in] generator The generator whose recurrent state is snapshotted.
 * \return OgaResult containing the error message if the snapshot failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_SnapshotState(OgaGenerator* generator);

/**
 * \brief Stages a hidden_states tensor as the model's hidden_states input for the next step.
 *        Used by the MTP self-speculative head, which consumes the main model's last hidden
 *        state. A no-op for models without a hidden_states input. The tensor is [batch,
 *        sequence_length, hidden_size] of the model's io dtype.
 * \param[in] generator The generator to set the hidden_states input on.
 * \param[in] hidden_states The hidden state values to feed on the next step.
 * \return OgaResult containing the error message if the call failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_SetHiddenStates(OgaGenerator* generator, OgaTensor* hidden_states);

/**
 * \brief Returns a copy of the model input identified by the given name as an OgaTensor on CPU. The buffer is owned by returned OgaTensor
 *       and will be released when the OgaTensor is destroyed
 * \param[in] generator The generator to run the GetInput on the name provided and the out pointer to store the input.
 * \param[in] name The name of the input tensor.
 * \param[out] out The returned OgaTensor.
 * \return OgaResult containing the error message if the computation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GetInput(const OgaGenerator* generator, const char* name, OgaTensor** out);

/**
 * \brief Returns a copy of the model output identified by the given name as an OgaTensor on CPU. The buffer is owned by returned OgaTensor
 *       and will be released when the OgaTensor is destroyed
 * \param[in] generator The generator to run the GetOutput on the name provided and the out pointer to store the output.
 * \param[in] name The name of the output tensor.
 * \param[out] out The returned OgaTensor.
 * \return OgaResult containing the error message if the computation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GetOutput(const OgaGenerator* generator, const char* name, OgaTensor** out);

/**
 * \brief Returns a copy of the logits from the model as an OgaTensor on CPU. The buffer is owned by returned OgaTensor
 *        and will be released when the OgaTensor is destroyed
 * \param[in] generator The generator get the logits from
 * \param[out] out The OgaTensor containing the logits, it only contains the last token logits even in prompt processing
 * \return OgaResult containing the error message if the computation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GetLogits(OgaGenerator* generator, OgaTensor** out);

/**
 * \brief Sets the logits to the generator. This is useful when the user wants to set the logits to a specific value
 *        for example when doing guided generation.
 * \param[in] generator The generator to set the logits on
 * \param[in] tensor The OgaTensor containing the logits, it must have the same shape as the logits returned by GetLogits
 * which is the last token logits.
 * \return OgaResult containing the error message if the setting of the logits failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_SetLogits(OgaGenerator* generator, OgaTensor* tensor);

/*
 * \brief Returns the number of tokens in the sequence at the given index.
 * \param[in] generator The generator to get the count of the tokens for the sequence at the given index.
 * \param[in] index The given index.
 * \return The number tokens in the sequence at the given index.
 */
OGA_EXPORT size_t OGA_API_CALL OgaGenerator_GetSequenceCount(const OgaGenerator* generator, size_t index);

/**
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaGenerator_GetSequenceCount
 * \param[in] generator The generator to get the sequence data for the sequence at the given index.
 * \param[in] index The given index.
 * \return The pointer to the sequence data at the given index. The sequence data is owned by the OgaGenerator
 *         and will be freed when the OgaGenerator is destroyed. The caller must copy the data if it needs to
 *         be used after the OgaGenerator is destroyed.
 */
OGA_EXPORT const int32_t* OGA_API_CALL OgaGenerator_GetSequenceData(const OgaGenerator* generator, size_t index);

/** \brief Creates an immutable snapshot of the accumulated speculative-decoding statistics.
 *  Values are zero for non-speculative models. The caller must destroy the returned object. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GetSpeculativeStats(const OgaGenerator* generator, OgaSpeculativeStats** out);
OGA_EXPORT void OGA_API_CALL OgaDestroySpeculativeStats(OgaSpeculativeStats* stats);

/** \brief Gets an integer counter from a speculative statistics snapshot. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSpeculativeStatsGetCount(
    const OgaSpeculativeStats* stats, const char* name, uint64_t* value);

/** \brief Gets the number of rounds that accepted exactly the requested number of draft tokens. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSpeculativeStatsGetAcceptanceLengthCount(
    const OgaSpeculativeStats* stats, size_t accepted_length, uint64_t* value);

/** \brief Gets the number of bins in the acceptance-length histogram. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSpeculativeStatsGetAcceptanceLengthHistogramSize(
    const OgaSpeculativeStats* stats, size_t* value);

/** \brief Gets a timing, rate, ratio, average, or speedup value from a statistics snapshot. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSpeculativeStatsGetNumber(
    const OgaSpeculativeStats* stats, const char* name, double* value);

/** \brief Gets a boolean value from a speculative statistics snapshot. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSpeculativeStatsGetBool(
    const OgaSpeculativeStats* stats, const char* name, bool* value);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerFromConfig(const OgaConfig* config, OgaTokenizer** out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerFromPath(const char* config_path, OgaTokenizer** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer*);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateMultiModalProcessor(const OgaModel* model, OgaMultiModalProcessor** out);

OGA_EXPORT void OGA_API_CALL OgaDestroyMultiModalProcessor(OgaMultiModalProcessor* processor);

/**
 * Updates tokenizer options for the given OgaTokenizer instance.
 * The provided keys and values must be null-terminated UTF-8 strings.
 *
 * This function allows updating tokenizer behavior at runtime by passing
 * key/value string pairs. Each key corresponds to a configurable tokenizer
 * option. Both keys and values must remain valid for the duration of this call.
 *
 * @param tokenizer Pointer to the OgaTokenizer whose options will be updated.
 * @param keys Array of option key strings.
 * @param values Array of corresponding option value strings (same length as keys).
 * @param num_options Number of key/value pairs provided.
 *
 * @return nullptr on success, or an OgaResult* describing the error.
 *         The returned OgaResult* (if not null) must be freed with OgaDestroyResult.
 *
 * Supported options:
 *
 * - `add_special_tokens`
 *   - Purpose: Controls whether to add special tokens (e.g., BOS/EOS) during tokenization.
 *   - Values: `"true"` / `"false"` or `"1"` / `"0"`.
 *   - Default: `"false"`. This is the default value set by ORT GenAI prior to any options updating.
 *
 * - `skip_special_tokens`
 *   - Purpose: Controls whether to remove special tokens during detokenization.
 *   - Values: `"true"` / `"false"` or `"1"` / `"0"`.
 *   - Default: `"true"`. This is the default value set by ORT GenAI prior to any options updating.
 *
 * - `chat_template_kwargs`
 *   - Purpose: Adds typed values to the chat template context.
 *   - Values: A C string containing serialized JSON text, such as `"{\"enable_thinking\":false}"`.
 *   - Default: `"{}"`. Set the value to `"{}"` to clear previously configured values.
 *
 * Future tokenizer options may be added without changing this API signature.
 * Passing unknown keys will result in an error.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaUpdateTokenizerOptions(
    OgaTokenizer* tokenizer,
    const char* const* keys,
    const char* const* values,
    size_t num_options);

/**
 * \brief Return the int representation of the BOS token
 * \param[in] tokenizer The tokenizer to read from
 * \param[out] token_id The BOS token id
 * \return OgaResult containing the error message if returning the BOS token id fails.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerGetBosTokenId(const OgaTokenizer* tokenizer, int32_t* token_id);

/**
 * \brief Return an array containing the int representations of the EOS token ids. The array is owned by the tokenizer and will be freed when the tokenizer is destroyed.
 * \param[in] tokenizer The tokenizer to read from
 * \param[out] eos_token_ids The array of EOS token ids
 * \param[out] token_count The length of the array
 * \return OgaResult containing the error message if returning the EOS token ids fails.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerGetEosTokenIds(const OgaTokenizer* tokenizer, const int32_t** eos_token_ids, size_t* token_count);

/**
 * \brief Return the int representation of the PAD token
 * \param[in] tokenizer The tokenizer to read from
 * \param[out] token_id The PAD token id
 * \return OgaResult containing the error message if returning the PAD token id fails.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerGetPadTokenId(const OgaTokenizer* tokenizer, int32_t* token_id);

/**
 * \brief Return the BOT (beginning of tool call) token id. Returns an error if the model does not define one.
 * \param[in] tokenizer The tokenizer to read from
 * \param[out] token_id The BOT token id
 * \return OgaResult containing the error message if the call fails.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerGetBotTokenId(const OgaTokenizer* tokenizer, int32_t* token_id);

/**
 * \brief Return the EOT (end of tool call) token id. Returns an error if the model does not define one.
 * \param[in] tokenizer The tokenizer to read from
 * \param[out] token_id The EOT token id
 * \return OgaResult containing the error message if the call fails.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerGetEotTokenId(const OgaTokenizer* tokenizer, int32_t* token_id);

/**
 * \brief Return the BOR (beginning of reasoning) token id. Returns an error if the model does not define one.
 * \param[in] tokenizer The tokenizer to read from
 * \param[out] token_id The BOR token id
 * \return OgaResult containing the error message if the call fails.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerGetBorTokenId(const OgaTokenizer* tokenizer, int32_t* token_id);

/**
 * \brief Return the EOR (end of reasoning) token id. Returns an error if the model does not define one.
 * \param[in] tokenizer The tokenizer to read from
 * \param[out] token_id The EOR token id
 * \return OgaResult containing the error message if the call fails.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerGetEorTokenId(const OgaTokenizer* tokenizer, int32_t* token_id);

/**
 * Encodes a single string and adds the encoded sequence of tokens to the OgaSequences. The OgaSequences must be freed with OgaDestroySequences
 * when it is no longer needed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer*, const char* str, OgaSequences* sequences);

/**
 * Batch encode an array of strings and return a single tensor output
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncodeBatch(const OgaTokenizer*, const char** strings, size_t count, OgaTensor** out);

/**
 * Batch decode a tensor of token ids and return an array of strings
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecodeBatch(const OgaTokenizer*, const OgaTensor* tensor, OgaStringArray** out);

/**
 * \brief Converts the given string to a single token id.
 * \param[in] tokenizer The tokenizer to use to convert the string to a token id.
 * \param[in] str The string to convert to a token id.
 * \param[in] token_id The converted token id.
 * \return OgaResult containing the error message if the conversion of the string to a token id failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerToTokenId(const OgaTokenizer* tokenizer, const char* str, int32_t* token_id);

/**
 * \brief Process images with input prompt
 * \param[in] processor The processor to use to process the images and prompt.
 * \param[in] prompt The prompt to use with the images.
 * \param[in] images The images to process.
 * \return OgaResult containing the named tensors for the processed inputs.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessImages(const OgaMultiModalProcessor*, const char* prompt, const OgaImages* images, OgaNamedTensors** input_tensors);

/**
 * \brief Process images with input prompts
 * \param[in] processor The processor to use to process the images and prompts.
 * \param[in] prompts The prompts to use with the images.
 * \param[in] images The images to process.
 * \return OgaResult containing the named tensors for the processed inputs.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessImagesAndPrompts(const OgaMultiModalProcessor*, const OgaStringArray* prompts, const OgaImages* images, OgaNamedTensors** input_tensors);

/**
 * \brief Process audios with input prompt
 * \param[in] processor The processor to use to process the audios and prompt.
 * \param[in] prompt The prompt to use with the audios.
 * \param[in] audios The audios to process.
 * \return OgaResult containing the named tensors for the processed inputs.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessAudios(const OgaMultiModalProcessor*, const char* prompt, const OgaAudios* audios, OgaNamedTensors** input_tensors);

/**
 * \brief Process audios with input prompts
 * \param[in] processor The processor to use to process the audios and prompts.
 * \param[in] prompts The prompts to use with the audios.
 * \param[in] audios The audios to process.
 * \return OgaResult containing the named tensors for the processed inputs.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessAudiosAndPrompts(const OgaMultiModalProcessor*, const OgaStringArray* prompts, const OgaAudios* audios, OgaNamedTensors** input_tensors);

/**
 * \brief Process images and/or audios with input prompt
 * \param[in] processor The processor to use to process the images, audios, and/or prompt.
 * \param[in] prompt The prompt to use with the images and/or audios.
 * \param[in] images The images to process.
 * \param[in] audios The audios to process.
 * \return OgaResult containing the named tensors for the processed inputs.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessImagesAndAudios(const OgaMultiModalProcessor*, const char* prompt, const OgaImages* images, const OgaAudios* audios, OgaNamedTensors** input_tensors);

/**
 * \brief Process images and/or audios with input prompts
 * \param[in] processor The processor to use to process the images, audios, and/or prompts.
 * \param[in] prompts The prompts to use with the images and/or audios.
 * \param[in] images The images to process.
 * \param[in] audios The audios to process.
 * \return OgaResult containing the named tensors for the processed inputs.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessImagesAndAudiosAndPrompts(const OgaMultiModalProcessor*, const OgaStringArray* prompts, const OgaImages* images, const OgaAudios* audios, OgaNamedTensors** input_tensors);

/** Decode a single token sequence and returns a null terminated utf8 string. out_string must be freed with OgaDestroyString
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer*, const int32_t* tokens, size_t token_count, const char** out_string);
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorDecode(const OgaMultiModalProcessor*, const int32_t* tokens, size_t token_count, const char** out_string);

/**
 * @brief Applies a chat template to input messages
 *
 * This function processes the specified template with the provided input using the
 * tokenizer, and outputs the resulting string. Optionally, it can include a
 * generation prompt in the output.
 *
 * \param[in] tokenizer OgaTokenizer used for template processing.
 * \param[in] template_str Null-terminated string representing the chat template. Use nullptr to fall back to the default chat template from the tokenizer config.
 * \param[in] messages Null-terminated string containing the input messages to be processed.
 * \param[in] tools Null-terminated string containing the chat function calls if any. Use nullptr if none.
 * \param[in] add_generation_prompt Indicates whether to add a generation prompt to the output.
 * \param[out] out_string Pointer to where the output will be stored. The returned pointer must be freed with OgaDestroyString
 * \return OgaResult* containing the error message if the function fails
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerApplyChatTemplate(const OgaTokenizer*, const char* template_str, const char* messages, const char* tools, bool add_generation_prompt, const char** out_string);

/** OgaTokenizerStream is to decoded token strings incrementally, one token at a time.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer*, OgaTokenizerStream** out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStreamFromProcessor(const OgaMultiModalProcessor*, OgaTokenizerStream** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream*);

/**
 * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
 * The caller is responsible for concatenating each chunk together to generate the complete result.
 * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream*, int32_t token, const char** out);

/** Create an OgaTensor from an optional user owned buffer. If a user owned buffer is supplied, the OgaTensor does
 * not own the memory (as it has no way to free it) so the 'data' parameter must be valid for the lifetime of the OgaTensor.
 *  If the 'data' parameter is nullptr, the OgaTensor will allocate its own memory.
 *
 * \param[in] data User supplied memory pointer, if non nullptr it must remain valid for lifetime of the OgaTensor
 * \param[in] shape_dims Pointer to array of int64_t values that define the tensor shape, example [1 20 30] would be equivalent to a C array of [1][20][30]
 * \param[in] shape_dims_count Count of elements in the shape_dims array
 * \param[in] element_type The data type that 'data' points to.
 * \param[out] out Writes the newly created OgaTensor into this, must be destroyed with OgaDestroyTensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTensorFromBuffer(void* data, const int64_t* shape_dims, size_t shape_dims_count, OgaElementType element_type, OgaTensor** out);

OGA_EXPORT void OGA_API_CALL OgaDestroyTensor(OgaTensor* tensor);

/** Get the OgaElementType of the data stored in the OgaTensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetType(OgaTensor*, OgaElementType* out);

/** Get the number of dimensions of the OgaTensor's shape, typically used to allocate a buffer of this size then calling OgaTensorGetShape with it
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetShapeRank(OgaTensor*, size_t* out);

/** Copies the shape dimensions into the shape_dims parameters. shape_dims_count must match the value returned by OgaTensorGetShapeRank
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetShape(OgaTensor*, int64_t* shape_dims, size_t shape_dims_count);

/** A pointer to the tensor data, it is typically cast into the actual data type of the tensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetData(OgaTensor*, void** out);

/** \brief Create an OgaNamedTensors
 * \param[out] out The created OgaNamedTensors
 * \return OgaResult containing the error message if the creation of the OgaNamedTensors failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateNamedTensors(OgaNamedTensors** out);

/** \brief Lookup a tensor in a NamedTensor set by name
 * \param[in] named_tensors The named tensors to search
 * \param[in] name The name of the tensor to find
 * \param[out] out The tensor with the given name
 * \return OgaResult containing the error message if the tensor with the given name could not be found.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaNamedTensorsGet(OgaNamedTensors* named_tensors, const char* name, OgaTensor** out);

/** \brief Set a tensor in a NamedTensor set by name
 * \param[in] named_tensors The named tensors to set the tensor
 * \param[in] name The name of the tensor to set
 * \param[in] tensor The tensor to set
 * \return OgaResult containing the error message if the tensor with the given name could not be set.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaNamedTensorsSet(OgaNamedTensors* named_tensors, const char* name, OgaTensor* tensor);

/** \brief Delete a tensor in a NamedTensor set by name
 * \param[in] named_tensors The named tensors to remove the tensor
 * \param[in] name The name of the tensor to remove
 * \return OgaResult containing the error message if the tensor with the given name could not be removed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaNamedTensorsDelete(OgaNamedTensors* named_tensors, const char* name);

/** \brief Get the number of tensors in the NamedTensors
 * \param[in] named_tensors The named tensors to get the count of the tensors
 * \param[out] out The number of tensors in the NamedTensors
 * \return OgaResult containing the error message if the getting of the count of the tensors failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaNamedTensorsCount(const OgaNamedTensors* named_tensors, size_t* out);

/** \brief Return an OgaStringArray of the names of the tensors in an OgaNamedTensors object
 * \param[in] named_tensors The named tensors to get the names of the tensors
 * \param[out] out The OgaStringArray containing the names of the tensors
 * \return OgaResult containing the error message if the getting of the names of the tensors failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaNamedTensorsGetNames(const OgaNamedTensors* named_tensors, OgaStringArray** out);

OGA_EXPORT OgaResult* OGA_API_CALL OgaSetCurrentGpuDeviceId(int device_id);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGetCurrentGpuDeviceId(int* device_id);

/**
 * \brief Creates an object of type OgaStringArray.
 * \return The result of the operation. If the operation is successful, a nullptr is returned.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateStringArray(OgaStringArray** out);

/**
 * \brief Creates an object of type OgaStringArray from the given strings.
 * \return The result of the operation. If the operation is successful, a nullptr is returned.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateStringArrayFromStrings(const char* const* strs, size_t count, OgaStringArray** out);

/**
 * \brief Destroys OgaStringArray.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyStringArray(OgaStringArray* string_array);

/**
 * \brief Adds the given string to the string_array.
 * \param[inout] string_array The string array to which the string is to be added
 * \param[in] str The string to be added to the string_array.
 * \return The result of the operation. If the operation is successful, a nullptr is returned.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaStringArrayAddString(OgaStringArray* string_array, const char* str);

/**
 * \brief Gets the number of strings in the string_array.
 * \param[in] string_array The OgaStringArray object to get the count of the strings.
 * \param[out] out The number of strings in the string_array.
 * \return The result of the operation. If the operation is successful, a nullptr is returned.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaStringArrayGetCount(const OgaStringArray* string_array, size_t* out);

/**
 * \brief Get a string from a string_array
 * \param[in] string_array The OgaStringArray object to get the string from.
 * \param[in] index The index of the string to get.
 * \return The string at the given index.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaStringArrayGetString(const OgaStringArray* string_array, size_t index, const char** out);

/**
 * \brief Creates the OgaAdapters object that manages the adapters.
          - The OgaAdapters object is used to load all the model adapters.
          - It is responsible for reference counting the loaded adapters.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateAdapters(const OgaModel* model, OgaAdapters** out);

/**
 * \brief Destroys the OgaAdapters object.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyAdapters(OgaAdapters* adapters);

/**
 * \brief Loads the model adapter from the given adapter file path and adapter name.
 * \param[in] adapters The OgaAdapters object to load the adapter.
 * \param[in] adapter_file_path The file path of the adapter to load.
 * \param[in] adapter_name A unique identifier for the adapter chosed by the function invoker.
 *                         This name is used for querying the adapter.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadAdapter(OgaAdapters* adapters, const char* adapter_file_path,
                                                  const char* adapter_name);

/**
 * \brief Unloads the adapter with the given identifier from the previosly loaded adapters.
          If the adapter is not found, or if it cannot be unloaded (when it is in use), an error is returned.
 * \param[in] adapters The OgaAdapters object to unload the adapter.
 * \param[in] adapter_name The name of the adapter to unload.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaUnloadAdapter(OgaAdapters* adapters, const char* adapter_name);

/**
 * \brief Sets the adapter with the given adapter name as active for the given OgaGenerator object.
 * \param[in] generator The OgaGenerator object to set the active adapter.
 * \param[in] adapters The OgaAdapters object that manages the model adapters.
 * \param[in] adapter_name The name of the adapter to set as active.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetActiveAdapter(OgaGenerator* generator, OgaAdapters* adapters,
                                                       const char* adapter_name);

/**
 * \brief Creates an OgaEngine object from the given model.
 *
 * The OgaEngine is responsible for managing and scheduling multiple requests, executing model inference,
 * and coordinating batching, caching, and resource management for efficient processing. This function
 * initializes a new engine instance using the provided model, allowing requests to be added, removed, and
 * processed through the engine's API. The engine must be destroyed with OgaDestroyEngine when no longer needed.
 *
 * On success, the Engine retains the underlying Model internally. The caller may release its
 * OgaModel handle with OgaDestroyModel immediately after this call; the Model remains alive until
 * the Engine and any other retaining objects are destroyed.
 *
 * \param[in] model The model to use for the engine. The handle must be valid for this call but need
 * not remain owned by the caller after successful Engine creation.
 * \param[out] out Pointer to the created engine instance. On success, *out will be set to the new engine object.
 * \return OgaResult containing the error message if the engine creation failed, or nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateEngine(OgaModel* model, OgaEngine** out);

/**
 * \brief Returns a caller-owned snapshot of the Engine's configured runtime capabilities.
 * This call must run on the Engine's owner thread.
 * \param[in] engine The Engine to inspect.
 * \param[out] out The capability snapshot. Destroy it with OgaDestroyEngineCapabilities.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineGetCapabilities(
    const OgaEngine* engine, OgaEngineCapabilities** out);

/**
 * \brief Returns the configured maximum batch size.
 * Static Engines without an explicit setting report the configured default of four. This value
 * does not account for lower operational limits imposed by a scheduler implementation.
 */
OGA_EXPORT size_t OGA_API_CALL OgaEngineCapabilitiesGetConfiguredMaxBatchSize(
    const OgaEngineCapabilities* capabilities);

/** \brief Returns the dynamic scheduler's configured token budget, or zero for static batching. */
OGA_EXPORT size_t OGA_API_CALL OgaEngineCapabilitiesGetMaxScheduledTokens(
    const OgaEngineCapabilities* capabilities);

/**
 * \brief Returns the maximum logical token length of one Request, or zero when unavailable.
 *
 * The value reflects the resolved target cache after runtime-profile selection and auxiliary
 * drafter allocations. It is an exclusive-use structural limit, not current free capacity. Zero
 * means the cache-backed ceiling is unsupported or unavailable, including for non-paged Engines.
 */
OGA_EXPORT uint64_t OGA_API_CALL OgaEngineCapabilitiesGetMaxRequestLength(
    const OgaEngineCapabilities* capabilities);

/** \brief Destroys an Engine capability snapshot. */
OGA_EXPORT void OGA_API_CALL OgaDestroyEngineCapabilities(
    OgaEngineCapabilities* capabilities);

/**
 * \brief Destroys the given engine.
 *
 * Destroying an engine closes every request bound to it. Surviving request handles remain valid
 * closed handles and must still be released with OgaDestroyRequest. A resident static batch is
 * released as shared Engine storage during teardown, not by independent per-Request deallocation.
 * \param[in] engine The engine to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyEngine(OgaEngine* engine);

/**
 * \brief Creates reusable event storage bound to one Engine.
 *
 * The Buffer allocates its event capacity once and does not retain the Engine. It may only be used
 * with the same live Engine on that Engine's owner thread. Destroying the Engine invalidates the
 * Buffer for OgaEngineRun, but the Buffer must still be released with
 * OgaDestroyEngineEventBuffer. Buffer access, Run, and destruction must be serialized; getters
 * must not race a Run or destruction.
 *
 * \param[in] engine The Engine to bind.
 * \param[in] capacity Maximum number of borrowed events exposed after one Run.
 * \param[out] out The caller-owned Buffer handle.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateEngineEventBuffer(
    OgaEngine* engine, size_t capacity, OgaEngineEventBuffer** out);

/** \brief Releases an Engine event Buffer and invalidates all of its borrowed views. */
OGA_EXPORT void OGA_API_CALL OgaDestroyEngineEventBuffer(
    OgaEngineEventBuffer* buffer);

/**
 * \brief Runs synchronous Engine progress into reusable Engine-owned-format storage.
 *
 * After the Buffer, Engine binding, and owner thread are validated, its previously borrowed event
 * and usage views are invalidated. A call drains retained events or executes at most one model
 * transaction, but never both. Overflow remains retained by the Engine for later calls. Capacity one provides
 * one-event-at-a-time behavior through this same operation. A successful partial prefill may
 * produce zero events while work remains pending.
 *
 * \param[in] engine The Engine to advance.
 * \param[in,out] buffer Reusable Buffer created for this Engine.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineRun(
    OgaEngine* engine, OgaEngineEventBuffer* buffer);

/**
 * \brief Returns the number of borrowed event views produced by the last successful Run.
 *
 * Returns zero for a null Buffer.
 */
OGA_EXPORT size_t OGA_API_CALL OgaEngineEventBufferGetCount(
    const OgaEngineEventBuffer* buffer);

/**
 * \brief Returns a borrowed event view at index.
 *
 * Returns null when buffer is null or index is out of bounds. The returned pointer is valid only
 * until the next OgaEngineRun using this Buffer or Buffer destruction.
 */
OGA_EXPORT const OgaEngineEvent* OGA_API_CALL OgaEngineEventBufferGet(
    const OgaEngineEventBuffer* buffer, size_t index);

/**
 * \brief Reads fields from a borrowed opaque Engine event.
 *
 * Each getter below returns an owned OgaResult on a null event or output pointer and null on
 * success. Scalar outputs are initialized to zero and pointer outputs to null before a null event
 * is rejected. Payload meaning is selected by OgaEngineEventFlags.
 *
 * OgaEngineEventGetRequest returns a const borrowed Request alias that must not be destroyed.
 * OgaEngineEventGetUsage returns a borrowed usage view. Both have the event view's lifetime.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineEventGetFlags(
    const OgaEngineEvent* event, OgaEngineEventFlags* out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineEventGetRequest(
    const OgaEngineEvent* event, const OgaRequest** out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineEventGetTurnId(
    const OgaEngineEvent* event, uint64_t* out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineEventGetToken(
    const OgaEngineEvent* event, int32_t* out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineEventGetFinishReason(
    const OgaEngineEvent* event, OgaFinishReason* out);
/**
 * \brief Reads the caller-facing index into the turn's stop-string list that completed a match.
 *
 * Writes -1 for every event whose finish reason is not OgaFinishReason_StopString, including
 * token-only, capacity, cancellation, failure, EOS, and length-limit terminal events.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineEventGetMatchedStopStringIndex(
    const OgaEngineEvent* event, int32_t* out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineEventGetErrorCode(
    const OgaEngineEvent* event, OgaErrorCode* out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineEventGetUsage(
    const OgaEngineEvent* event, const OgaTurnUsage** out);

/** \brief Reads counters from a borrowed opaque Turn usage view. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnUsageGetPromptTokens(
    const OgaTurnUsage* usage, uint64_t* out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnUsageGetGeneratedTokens(
    const OgaTurnUsage* usage, uint64_t* out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnUsageGetCachedPromptTokens(
    const OgaTurnUsage* usage, uint64_t* out);

/**
 * \brief Checks if the engine has any pending requests to process.
 *
 * This owner-thread operation first reclaims Requests whose final public handle was released, then
 * reports whether undelivered events or schedulable work remain.
 * A false result does not mean the engine owns no requests: turn-complete requests remain owned
 * until closed or abandoned.
 *
 * \param[in] engine The engine instance to check for pending requests.
 * \param[out] out Pointer to a boolean value that will be set to true if there are pending requests, or false otherwise.
 * \return OgaResult containing the error message if the operation failed, or nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineHasPendingRequests(OgaEngine* engine, bool* out);

/**
 * \brief Creates a Request permanently bound to an Engine.
 *
 * A Request owns resident-session policy only. Generation policy belongs to each Turn, so nothing
 * about sampling, guidance, or stop strings is fixed here. Creation does not queue work; call
 * OgaRequestBeginTurn.
 *
 * \param[in] engine The owning Engine.
 * \param[in] options Nullable request-scoped options. Null options, or zero max_session_tokens, use
 * model-configured search.max_length capped by the Engine's max_request_length when that capability
 * is nonzero. A zero capability means the ceiling is unavailable and preserves search.max_length.
 * \param[out] out The caller-owned Request handle.
 * \return OgaResult containing the error message if the operation failed, or nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineCreateRequest(
    OgaEngine* engine, const OgaRequestOptions* options, OgaRequest** out);

/**
 * \brief Creates reusable Request options.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateRequestOptions(
    OgaRequestOptions** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyRequestOptions(
    OgaRequestOptions* options);
/**
 * \brief Sets the total tokens (prompt plus generated, across every Turn) the Request may reach.
 *
 * Zero restores the default, which is model-configured search.max_length capped by the Engine's
 * max_request_length when that capability is nonzero. A nonzero value may not exceed the capability
 * when available; otherwise it may not exceed search.max_length. This is the Request's one session
 * limit: Search completion, cache sizing, speculative bounds, and the MaxSessionTokens finish reason
 * all use it.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRequestOptionsSetMaxSessionTokens(
    OgaRequestOptions* options, uint64_t max_session_tokens);

/**
 * \brief Creates reusable Turn options bound to one Request.
 *
 * Every option below is unset by default, and an unset option means "use the model-configured
 * default for this Turn" -- never "keep what the previous Turn used". Seed is the one exception:
 * an unset seed continues the Request's existing random streams. A reused options object reapplies
 * its configured fields to every Turn it is passed to; OgaTurnOptionsReset removes them all.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRequestCreateTurnOptions(
    OgaRequest* request, OgaTurnOptions** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTurnOptions(OgaTurnOptions* options);
/** \brief Caps the tokens this Turn generates. Zero unsets the cap. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetMaxGeneratedTokens(
    OgaTurnOptions* options, uint64_t max_generated_tokens);
/**
 * \brief Masks the end-of-sequence token until this Turn has generated this many tokens.
 *
 * Zero unsets the minimum. It does not prevent stop strings, Turn or session limits, cancellation,
 * failure, or guidance termination. If guidance permits EOS and no continuation token, its
 * termination takes precedence rather than leaving the Turn with no legal token. Admission rejects
 * a minimum that exceeds the Turn's maximum or does not fit inside the Request's session limit.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetMinGeneratedTokens(
    OgaTurnOptions* options, uint64_t min_generated_tokens);
/** \brief Selects random sampling (true) or the top logit (false) for this Turn. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetDoSample(
    OgaTurnOptions* options, bool do_sample);
/**
 * \brief Sets this Turn's sampling temperature.
 *
 * Zero requests top-logit selection. Admission rejects an explicitly set temperature that the
 * resolved policy would ignore because the Turn selects the top logit for another reason.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetTemperature(
    OgaTurnOptions* options, float temperature);
/**
 * \brief Sets this Turn's nucleus (top-p) bound, between 0.0 and 1.0.
 *
 * Admission rejects an explicitly set top_p that the resolved policy would ignore.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetTopP(
    OgaTurnOptions* options, float top_p);
/**
 * \brief Sets this Turn's top-k bound. One requests top-logit selection; zero disables top-k.
 *
 * Admission rejects an explicitly set top_k that the resolved policy would ignore, and a sampled
 * Turn rejects a top_k greater than the model vocabulary size.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetTopK(
    OgaTurnOptions* options, int32_t top_k);
/** \brief Sets this Turn's repetition penalty. Must be finite and greater than zero. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetRepetitionPenalty(
    OgaTurnOptions* options, float repetition_penalty);
/**
 * \brief Forbids repeating any n-gram of this size in the generated sequence. Zero disables it.
 *
 * Admission rejects a nonzero size on a scoring device whose search cannot apply it, rather than
 * failing after the model has already run.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetNoRepeatNgramSize(
    OgaTurnOptions* options, int32_t no_repeat_ngram_size);
/**
 * \brief Reseeds the Request's random streams at the start of this Turn.
 *
 * Zero is a valid deterministic seed, so use OgaTurnOptionsClearSeed to remove a pending reseed.
 * Reusing a seeded options object deliberately reseeds identically at the start of every Turn it is
 * passed to. The seed takes effect only when the Turn's first sampling step commits, so a rolled
 * back step reseeds the retry identically. Requires an Engine configured for dynamic batching and,
 * when a device batched sampler is present, support for checkpointing its RNG state.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetSeed(
    OgaTurnOptions* options, uint64_t seed);
/**
 * \brief Removes a pending reseed, continuing the Request's existing random streams.
 *
 * This is not "randomize again": the streams simply carry on from where the previous Turn left off.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsClearSeed(
    OgaTurnOptions* options);
/**
 * \brief Sets decoded UTF-8 stop strings for a Turn, copying them immediately.
 *
 * Reusing or destroying stop_strings after this call cannot affect the options. An empty
 * stop_strings array (zero entries) clears/disables stop strings on this options object. This is
 * distinct from a nonempty array containing an empty string member: every entry in a nonempty array
 * must itself be a nonempty, valid UTF-8 string, or this call fails and the prior configuration is
 * left unchanged. The configuration as a whole may contain at most 16 entries totaling at most 16
 * KiB. Duplicate entries are preserved as distinct, independently indexed entries. Matching is exact:
 * no normalization, trimming, or case folding, and only text this Engine Request generates during
 * the turn (never prompt or earlier-turn tokens) is considered.
 *
 * A subsequent OgaRequestBeginTurn snapshots these options, so reusing or mutating this
 * OgaTurnOptions afterward cannot alter an already-active turn.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetStopStrings(
    OgaTurnOptions* options, const OgaStringArray* stop_strings);
/**
 * \brief Constrains this Turn's output to a grammar, copying both strings immediately.
 *
 * guidance_type is one of "json_schema", "regex", or "lark_grammar". The grammar is compiled and
 * validated before the Turn mutates the Request, so an invalid grammar leaves the Request exactly
 * as it was. Guidance is strictly Turn-scoped: a following Turn that sets none is unguided. A
 * guided Turn does not accept speculative drafts; the next unguided Turn does again.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsSetGuidance(
    OgaTurnOptions* options, const char* guidance_type,
    const char* guidance_data);
/** \brief Removes the configured grammar, so the Turn is unguided. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsClearGuidance(
    OgaTurnOptions* options);
/** \brief Restores every Turn option to its unset state. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTurnOptionsReset(
    OgaTurnOptions* options);

/** \brief Begins a Turn, copying input and supported options before return. */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRequestBeginTurn(
    OgaRequest* request, const OgaTurnOptions* turn_options,
    const int32_t* input_ids, uint64_t input_ids_count,
    uint64_t* out_turn_id);

/**
 * \brief Cancels the named queued or active Turn.
 *
 * Cancellation is synchronous and owner-thread-only. A matching queued or active Turn produces one
 * terminal Cancelled event. Stale, unknown, and completed IDs return success with false. A Request
 * that has never begun a Turn or has already been closed returns an error.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRequestCancelTurn(
    OgaRequest* request, uint64_t turn_id, bool* out_cancelled);

/**
 * \brief Rewinds a completed Request to the sequence boundary before a Turn began.
 *
 * Rewind is synchronous and owner-thread-only. The current Turn must be complete, its events must
 * have been drained, and its finish reason must not be Failed. turn_id must identify a Turn
 * previously begun by this Request and still present in its active history. The named Turn and all
 * later Turns are discarded. Turn IDs are never reused, and sampling continues from its current
 * random stream rather than restoring a previous sampling state. The operation releases the
 * Request's model-state ownership, including auxiliary MTP or DFlash/DSpark state; indexed prefix
 * blocks may remain cached. Another rewind to an earlier retained Turn is allowed before the next
 * OgaRequestBeginTurn even though the Request is no longer resident. That call creates a new Turn
 * and replays the retained prefix together with its new input before generation, possibly adopting
 * matching cached blocks.
 *
 * Dynamic paged Requests release their paged blocks and any fixed recurrent/convolution slot
 * atomically. Static Requests are rewindable only when they are the sole resident row, because one
 * row cannot be removed from a shared contiguous allocation. No event is emitted by rewind.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRequestRewindToStartOfTurn(
    OgaRequest* request, uint64_t turn_id);

/**
 * \brief Permanently closes a Request and releases its Engine resources.
 *
 * Close is valid from every lifecycle state and is idempotent. It logically removes the Request
 * from scheduling and discards its undelivered events. Dynamic cache ownership is released
 * immediately. A resident static-batch row and its shared cache allocation may remain physically
 * retained until the whole batch is recycled.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRequestClose(OgaRequest* request);

/**
 * \brief Proposes speculative draft tokens for the request's next decode operation.
 *
 * The request must be ready to decode. The operation then runs one row per draft on top of its own
 * token, verifies each draft against the model's prediction, and keeps the accepted prefix. The
 * proposal applies to the next step only:
 * a committed step consumes it whether or not it could verify it. Passing an empty sequence clears
 * the proposal. Random target sampling is supported for deterministic draft proposals when top_k
 * is positive. Requires an engine whose cache can roll a rejected draft back (see
 * OgaEngineMaxDraftTokensPerProposal).
 * Seeded sampled output is reproducible only when both the proposal and scheduling path are the
 * same, because draft admission changes which random stream performs target sampling.
 *
 * \param[in] request The request to propose drafts for.
 * \param[in] tokens One sequence holding the draft continuation, in order.
 * \return OgaResult containing the error message if the proposal was rejected, or nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRequestSetDraftTokens(OgaRequest* request, const OgaSequences* tokens);

/**
 * \brief Reports how many speculative draft tokens one request may attach to a proposal.
 *
 * Zero means this engine cannot roll a rejected draft back, so OgaRequestSetDraftTokens will fail.
 * This query must be called from the Engine owner thread.
 *
 * \param[in] engine The engine to query.
 * \param[out] out The maximum draft token count.
 * \return OgaResult containing the error message on failure, or nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineMaxDraftTokensPerProposal(const OgaEngine* engine, size_t* out);

/**
 * \brief Returns a cumulative snapshot of Engine speculative-decoding statistics.
 *
 * Target and draft forward-pass counters include successful model executions even if a later
 * transaction phase rolls back. Proposal, acceptance, and acceptance-length counters include only
 * committed speculative steps.
 *
 * \param[in] engine The engine to query.
 * \param[out] out The newly allocated statistics snapshot. Destroy it with OgaDestroySpeculativeStats.
 * \return OgaResult containing the error message on failure, or nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaEngineGetSpeculativeStats(
    const OgaEngine* engine, OgaSpeculativeStats** out);

/**
 * \brief Destroys the given request.
 *
 * Releasing the final owned handle without first calling OgaRequestClose marks the Request
 * abandoned for reclamation at the next owner-thread Engine boundary. This final handle release
 * may occur on another thread because the Engine strongly retains the Request and the release only
 * publishes an atomic marker. Reclamation has the same logical scheduling, event-purge, and
 * runtime-state release behavior as Close. A resident static-batch row and shared cache allocation
 * may still remain until the batch is recycled.
 *
 * \param[in] request A Request handle returned by OgaEngineCreateRequest.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyRequest(OgaRequest* request);

/**
 * \brief Registers an execution provider library with ONNXRuntime API.
 * \param registration_name name for registration.
 * \param path provider path.
 *
 */
OGA_EXPORT void OGA_API_CALL OgaRegisterExecutionProviderLibrary(const char* registration_name, const char* library_path);

/**
 * \brief Unregisters an execution provider library with ONNXRuntime API.
 * \param registration_name name for registration.
 *
 */
OGA_EXPORT void OGA_API_CALL OgaUnregisterExecutionProviderLibrary(const char* registration_name);

/**
 * \brief Creates a StreamingProcessor for mel spectrogram extraction from raw audio.
 * \param[in] model The model to create the processor for (must be nemotron_speech type).
 * \param[out] out Pointer to store the created StreamingProcessor instance.
 * \return OgaResult on error, nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateStreamingProcessor(OgaModel* model, OgaStreamingProcessor** out);

/**
 * \brief Process a chunk of raw PCM audio and return a NamedTensors if a full chunk is ready.
 * \param[in] processor The StreamingProcessor instance.
 * \param[in] audio_data Pointer to float32 PCM audio samples (mono, model sample rate).
 * \param[in] num_samples Number of audio samples.
 * \param[out] out Pointer to store the NamedTensors. Set to nullptr if not enough audio yet.
 *                  Caller must free with OgaDestroyNamedTensors.
 * \return OgaResult on error, nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaStreamingProcessorProcess(OgaStreamingProcessor* processor, const float* audio_data, size_t num_samples, OgaNamedTensors** out);

/**
 * \brief Flush remaining buffered audio (pads with silence).
 * \param[in] processor The StreamingProcessor instance.
 * \param[out] out Pointer to store the NamedTensors. Set to nullptr if buffer was empty.
 * \return OgaResult on error, nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaStreamingProcessorFlush(OgaStreamingProcessor* processor, OgaNamedTensors** out);

/**
 * \brief Destroy a StreamingProcessor instance.
 * \param[in] processor The StreamingProcessor instance to destroy.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyStreamingProcessor(OgaStreamingProcessor* processor);

/**
 * \brief Set a processor option as a key-value pair.
 *        Supported keys: "use_vad", "vad_threshold", "silence_duration_ms", "prefix_padding_ms".
 * \param[in] processor The StreamingProcessor instance.
 * \param[in] key Option name.
 * \param[in] value Option value as string.
 * \return OgaResult on error, nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaStreamingProcessorSetOption(OgaStreamingProcessor* processor, const char* key, const char* value);

/**
 * \brief Get a processor option value by key.
 * \param[in] processor The StreamingProcessor instance.
 * \param[in] key Option name.
 * \param[out] value Pointer to store the value string. Caller must free with OgaDestroyString.
 * \return OgaResult on error, nullptr on success.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaStreamingProcessorGetOption(const OgaStreamingProcessor* processor, const char* key, const char** value);

#ifdef __cplusplus
}
#endif
//! @}
