/*
 * File: onnxInference.c
 * Description:
 *    Implementation of helper functions for running ONNX models using ONNX Runtime.
 *    This file contains routines for creating inference pipelines, managing input/output tensors,
 *    and building prototypes.
 *
 *    Designed to enable quick, modular, plug-and-play prototyping of inference workflows
 * 
 * Author: Shamith Achanta
 */

#include "onnxInference.h"

// Global OrtApi instance
OrtGlobalState g_state;

/* Helper Functions */
char* path_join(const char* path1, const char* path2) {
  // Calculate the length of both input paths
  size_t len1 = strlen(path1);
  size_t len2 = strlen(path2);

  // Ensure space for the combined path and the null terminator
  size_t total_len = len1 + len2 + 2;

  // Allocate memory for the combined path
  char* result = (char*) malloc(total_len * sizeof(char));
  if (!result) return NULL;

  // Copy the first path into the result
  strcpy(result, path1);

  // Add a path separator
  #ifdef __unix__
    if (result[len1 - 1] != '/') strcat(result, "/");
  #else
    if (result[len1 - 1] != '\\') strcat(result, "\\");
  #endif

  // Append the second path
  strcat(result, path2);

  return result;
}

// Function to create ORTSessionOptions
ORTSessionOptions* initSessionOptions(
  const char* execution_provider, const char* memory_device,
  DynamicAxes* dynamic_axes, int intra_threads, int inter_threads,
  int graph_optimization_level, int execution_mode, int execution_order, int log_severity_level) {
  
  ORTSessionOptions* options = (ORTSessionOptions*) malloc(sizeof(ORTSessionOptions));
  if (!options) return NULL;  // Handle allocation failure

  options->dynamic_axes = dynamic_axes;
  options->execution_provider = strdup(execution_provider);
  options->memory_device = strdup(memory_device);
  options->intra_threads = intra_threads;
  options->inter_threads = inter_threads;
  options->graph_optimization_level = graph_optimization_level;
  options->execution_mode = execution_mode;
  options->execution_order = execution_order;
  options->log_severity_level = log_severity_level;
  
  return options;
}

// Function to free ORTSessionOptions
void freeSessionOptions(ORTSessionOptions* options) {
  if (options) {
    free((char*) options->execution_provider);
    free((char*) options->memory_device);
    free(options);
  }
}

// Get element size from data type
size_t GetElementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return sizeof(int8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return sizeof(int16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return sizeof(int64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return sizeof(char*);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return sizeof(bool);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(float) / 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return sizeof(double);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return sizeof(uint32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return sizeof(uint64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return sizeof(float) / 2;
    default: return 0;
  }
}

// Get data type of Tensor
const char* get_tensor_element_type_string(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64_t";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "complex64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "complex128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
    default: return "unknown";
  }
}

// Initialize the ONNX Runtime API
bool ORT_Initialize(void) {
  if (g_state.initialized) return true;

  // Get API base
  g_state.api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_state.api) {
    LOG_MESSAGE(LOG_ERROR, "Failed to initialize ONNX Runtime API version: %d", ORT_API_VERSION);
    return false;
  }

  // Create environment
  OrtStatus* status = g_state.api->CreateEnv(
      ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference", &g_state.env);
  if (status != NULL) {
    ORT_CHECK_STATUS(status, "Cannot create ONNX Runtime environment",
      do { g_state.api = NULL; } while (0), false);
  }

  // Get default allocator
  status = g_state.api->GetAllocatorWithDefaultOptions(&g_state.allocator);
  if (status != NULL) {
    ORT_CHECK_STATUS(status, "Cannot get default allocator",
      do { g_state.api->ReleaseEnv(g_state.env); g_state.env = NULL; g_state.api = NULL; } while (0), false);
  }

  g_state.initialized = true;
  return true;
}

// Check if ORT is initialized
bool ORT_IsInitialized(void) {
  return g_state.initialized;
}

// Cleanup and release all resources
void ORT_Cleanup(void) {
  if (!g_state.initialized) return;

  if (g_state.env) g_state.api->ReleaseEnv(g_state.env); g_state.env = NULL;

  // Reset global state
  g_state.allocator = NULL; g_state.api = NULL; g_state.initialized = false;
}

// Create a tensor with specified shape and data
OrtValue* createTensor(const void* data, const int64_t* shape,
                           size_t dim_count,
                           ONNXTensorElementDataType element_type) {
  if (!data || !shape || dim_count == 0) {
    LOG_MESSAGE(LOG_ERROR, "Invalid tensor parameters");
    return NULL;
  }

  if (!ORT_IsInitialized()) ORT_Initialize();

  // Calculate total element count
  size_t element_count = 1;
  for (size_t i = 0; i < dim_count; i++) {
    if (shape[i] <= 0) {
      LOG_MESSAGE(LOG_ERROR, "Invalid dimension size at index %zu: %lld", i, (long long) shape[i]);
      return NULL;
    }
    element_count *= shape[i];
  }

  // Get element size
  size_t element_size = GetElementSize(element_type);
  if (!element_size) {
    LOG_MESSAGE(LOG_ERROR, "Invalid element type: %d", element_type);
    return NULL;
  }

  // Calculate total tensor size in bytes
  size_t tensor_size = element_count * element_size;

  // Create memory info
  OrtMemoryInfo* memory_info = NULL;
  OrtStatus* status = g_state.api->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  if (status != NULL) ORT_CHECK_STATUS(status, "Failed to create memory info", NULL, NULL);

  // Create tensor
  OrtValue* tensor = NULL;
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    OrtAllocator* allocator = NULL;
    status = g_state.api->GetAllocatorWithDefaultOptions(&allocator);

    status = g_state.api->CreateTensorAsOrtValue(
        allocator, shape, dim_count, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &tensor);

    status = g_state.api->FillStringTensor(tensor, (const char * const*) data, element_count);
  }

  else {
    status = g_state.api->CreateTensorWithDataAsOrtValue(
        memory_info, (void*) data, tensor_size, shape, dim_count, element_type,
        &tensor);
  }

  // Release memory info
  g_state.api->ReleaseMemoryInfo(memory_info);

  if (status != NULL) ORT_CHECK_STATUS(status, "Failed to create tensor", NULL, NULL);

  return tensor;
}

// Release a tensor
void freeTensor(OrtValue* tensor) {
  if (tensor) g_state.api->ReleaseValue(tensor);
}

// Get Tensor information
static void getTensorInfo(const OrtTensorTypeAndShapeInfo* tensor_info, TensorInfo* info) {
  if (!tensor_info || !info) return;

  OrtStatus* status = NULL;

  status = g_state.api->GetTensorElementType(tensor_info, &info->dtype);
  if (status != NULL) ORT_CHECK_STATUS_VOID(status, "GetTensorElementType failed", NULL);

  status = g_state.api->GetDimensionsCount(tensor_info, &info->ndims);
  if (status != NULL) ORT_CHECK_STATUS_VOID(status, "GetDimensionsCount failed", NULL);

  info->dimensions = (int64_t*) malloc(info->ndims * sizeof(int64_t));
  if (!info->dimensions) ORT_CHECK_STATUS_VOID(status, "Failed to allocate memory for dimensions", NULL);

  status = g_state.api->GetDimensions(tensor_info, info->dimensions, info->ndims);
  if (status != NULL) ORT_CHECK_STATUS_VOID(status, "GetDimensions failed", NULL);

  status = g_state.api->GetTensorShapeElementCount(tensor_info, &info->nelements);
  if (status != NULL) info->nelements = 0;
}

// Free tensor info resources
void freeTensorInfo(TensorInfo* info) {
  if (info && info->dimensions) free(info->dimensions); free(info);
}

// Print tensor info to any file or string buffer
void printTensorInfo(TensorInfo* info, FILE* stream) {
  if (!info || !stream) return;

  fprintf(stream, "Tensor: shape: [");
  for (size_t i = 0; i < info->ndims; i++) {
    fprintf(stream, "%ld", (long) info->dimensions[i]);
    if (i < info->ndims - 1) fprintf(stream, ", ");
  }

  fprintf(stream, "], dtype: %s\n", get_tensor_element_type_string(info->dtype));
}

// Get Tensor Info
void inspectTensor(OrtValue* tensor, bool verbose, size_t num_elements) {
  if (!tensor) {
    LOG_MESSAGE(LOG_ERROR, "Invalid tensor passed as argument");
    return;
  }

  OrtTensorTypeAndShapeInfo* tensor_info = NULL;
  TensorInfo* info = (TensorInfo*) malloc(sizeof(TensorInfo));

  OrtStatus* status = g_state.api->GetTensorTypeAndShape(tensor, &tensor_info);

  getTensorInfo(tensor_info, info);
  g_state.api->ReleaseTensorTypeAndShapeInfo(tensor_info);

  printTensorInfo(info, stdout);

  // Get data pointer
  void* data = NULL;
  status = g_state.api->GetTensorMutableData(tensor, &data);

  // Print first n elements
  if (verbose && info->nelements > 0) {
    size_t num_elements_to_print = info->nelements < num_elements ? info->nelements : num_elements;
    printf("\tdata (first %zu elements): ", num_elements_to_print);
    switch (info->dtype) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
        float* float_data = (float*) data;
        for (size_t i = 0; i < num_elements_to_print; i++) {
          printf("%f ", float_data[i]);
        }
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
        int32_t* int_data = (int32_t*) data;
        for (size_t i = 0; i < num_elements_to_print; i++) {
          printf("%d ", int_data[i]);
        }
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
        int64_t* int_data = (int64_t*) data;
        for (size_t i = 0; i < num_elements_to_print; i++) {
          printf("%lld ", (long long) int_data[i]);
        }
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
        size_t string_len = 0;

        for (size_t i = 0; i < num_elements_to_print; i++) {
          status = g_state.api->GetStringTensorElementLength(tensor, i, &string_len);
          if (status != NULL) ORT_CHECK_STATUS_VOID(status, "GetStringTensorElementLength failed", NULL);

          char* string_data = (char*) malloc(string_len + 1);
          status = g_state.api->GetStringTensorElement(tensor, string_len, i, string_data);
          string_data[string_len] = '\0'; // null-terminate string
          if (status != NULL) ORT_CHECK_STATUS_VOID(status, "GetStringTensorElement failed: %s", free(string_data));

          printf("%s ", string_data); free(string_data);
        }
        break;
      }
      default:
        printf("(Unsupported type for preview)");
        break;
    }

    printf("\n");
  }

  freeTensorInfo(info);
}

// Print model info to any file or string buffer
void printModelInfo(ModelInfo* info, FILE* stream) {
  if (!info || !stream) return;

  fprintf(stream, "Model Inputs (%zu):\n", info->input_count);
  for (size_t i = 0; i < info->input_count; i++) {
    printf("\t"); printTensorInfo(&info->inputs[i], stream);
  }

  fprintf(stream, "Model Outputs (%zu):\n", info->output_count);
  for (size_t i = 0; i < info->output_count; i++) {
    printf("\t"); printTensorInfo(&info->outputs[i], stream);
  }
}

// Get ONNX model inputs and outputs shape
void inspectModel(OrtSession* session) {
  if (!session) {
    LOG_MESSAGE(LOG_ERROR, "Invalid session provided to inspectModel");
    return;
  }

  OrtStatus* status = NULL;
  OrtAllocator* allocator = NULL;

  status = g_state.api->GetAllocatorWithDefaultOptions(&allocator);
  if (status != NULL) ORT_CHECK_STATUS_VOID(status, "GetAllocatorWithDefaultOptions failed", NULL);

  ModelInfo* model_info = (ModelInfo*) calloc(1, sizeof(ModelInfo));
  if (!model_info) ORT_CHECK_STATUS_VOID(status, "Cannot allocate memory for ModelInfo", NULL);

  // Get input count
  status = g_state.api->SessionGetInputCount(session, &model_info->input_count);
  if (status != NULL) ORT_CHECK_STATUS_VOID(status, "SessionGetInputCount failed", freeModelInfo(model_info));

  // Allocate input info array
  model_info->inputs =
      (TensorInfo*) calloc(model_info->input_count, sizeof(TensorInfo));
  if (!model_info->inputs) ORT_CHECK_STATUS_VOID(status, "Cannot alloc memory for model_info->inputs", freeModelInfo(model_info));

  // Get input info
  for (size_t i = 0; i < model_info->input_count; i++) {
    OrtTypeInfo* type_info = NULL;
    const OrtTensorTypeAndShapeInfo* tensor_info = NULL;

    status = g_state.api->SessionGetInputTypeInfo(session, i, &type_info);
    if (status != NULL) ORT_CHECK_STATUS_VOID(status, "SessionGetInputTypeInfo failed", freeModelInfo(model_info));

    status = g_state.api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (status != NULL) ORT_CHECK_STATUS_VOID(status, "CastTypeInfoToTensorInfo failed",
      do { freeModelInfo(model_info); g_state.api->ReleaseTypeInfo(type_info); } while(0));

    getTensorInfo(tensor_info, &model_info->inputs[i]);
    g_state.api->ReleaseTypeInfo(type_info);
  }

  // Get output count
  status = g_state.api->SessionGetOutputCount(session, &model_info->output_count);
  if (status != NULL) ORT_CHECK_STATUS_VOID(status, "SessionGetOutputCount failed", freeModelInfo(model_info));

  // Allocate output info array
  model_info->outputs = (TensorInfo*) calloc(model_info->output_count, sizeof(TensorInfo));
  if (!model_info->outputs) ORT_CHECK_STATUS_VOID(status, "Cannot alloc memory for model_info->outputs", freeModelInfo(model_info));

  // Get output info
  for (size_t i = 0; i < model_info->output_count; i++) {
    OrtTypeInfo* type_info = NULL;
    const OrtTensorTypeAndShapeInfo* tensor_info = NULL;

    status = g_state.api->SessionGetOutputTypeInfo(session, i, &type_info);
    if (status != NULL) ORT_CHECK_STATUS_VOID(status, "SessionGetOutputTypeInfo failed", freeModelInfo(model_info));

    status = g_state.api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (status != NULL) ORT_CHECK_STATUS_VOID(status, "CastTypeInfoToTensorInfo failed",
      do { freeModelInfo(model_info); g_state.api->ReleaseTypeInfo(type_info); } while(0));

    getTensorInfo(tensor_info, &model_info->outputs[i]);
    g_state.api->ReleaseTypeInfo(type_info);
  }

  printModelInfo(model_info, stdout);
  freeModelInfo(model_info);
}

// Free model info resources
void freeModelInfo(ModelInfo* info) {
  if (!info) return;

  for (size_t i = 0; i < info->input_count; i++) {
    freeTensorInfo(&info->inputs[i]);
  }

  for (size_t i = 0; i < info->output_count; i++) {
    freeTensorInfo(&info->outputs[i]);
  }

  free(info);
}

// Make dynamic axes static
static void set_dynamic_dimensions(OrtSessionOptions* session_options, const DynamicAxes* dynamic_axes) {
  if (!dynamic_axes) LOG_MESSAGE(LOG_ERROR, "Invalid arguments to set_dynamic_dimensions"); return;

  OrtStatus* status = NULL;

  // Iterate through dynamic axes and set them
  for (size_t i = 0; i < dynamic_axes->count; i++) {
    status = g_state.api->AddFreeDimensionOverrideByName(session_options, dynamic_axes->dims[i].name, dynamic_axes->dims[i].value);
    if (status!= NULL) ORT_CHECK_STATUS_VOID(status, "AddFreeDimensionOverrideByName failed", NULL);
  }

  if (status) g_state.api->ReleaseStatus(status);
}

// Path to ORT Extensions library
static const char* ort_extensions_lib_path(void) {
  #ifdef __unix__
    // int success = system("cd onnxruntime-extensions-" ORT_EXTENSIONS_VERSION " && CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_FLAVOR=RelWithDebInfo ./build.sh && cd ..");
    return "onnxruntime-extensions-" ORT_EXTENSIONS_VERSION "/out/Linux/RelWithDebInfo/lib/libortextensions.so";
  #else
    return "onnxruntime-extensions-" ORT_EXTENSIONS_VERSION "\\out\\Windows\\RelWithDebInfo\\lib\\ortextensions.dll";
  #endif
}

// Initialize ONNX Runtime inference session
OrtSession* initInference(const char* onnx_model_path,
                          const ORTSessionOptions* session_options) {
  if (!onnx_model_path || !session_options) {
    LOG_MESSAGE(LOG_ERROR, "Invalid parameters to initInference");
    return NULL;
  }

  // Initialize the ONNX Runtime API if not already done
  if (!ORT_IsInitialized()) ORT_Initialize();

  // Create session options
  OrtSessionOptions* ort_session_options;
  OrtStatus* status = g_state.api->CreateSessionOptions(&ort_session_options);
  if (status != NULL) {
    LOG_MESSAGE(LOG_ERROR, "Failed to create session options: %s", g_state.api->GetErrorMessage(status));
    g_state.api->ReleaseStatus(status);
    return NULL;
  }

  // Configure session options
  status = g_state.api->SetSessionLogSeverityLevel(
      ort_session_options, session_options->log_severity_level);
  ORT_CHECK_STATUS(
      status, "Failed to set log severity level",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Set log verbosity level using sessionoptions->log_severity_level
  status = g_state.api->SetSessionLogVerbosityLevel(
      ort_session_options, session_options->log_severity_level);
  ORT_CHECK_STATUS(
      status, "Failed to set log verbosity level",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Memory settings - CPU memory arena
  status = g_state.api->EnableCpuMemArena(ort_session_options);
  ORT_CHECK_STATUS(
      status, "Failed to enable CPU memory arena",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Memory pattern
  status = g_state.api->EnableMemPattern(ort_session_options);
  ORT_CHECK_STATUS(
      status, "Failed to enable memory pattern",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Thread settings
  status = g_state.api->SetIntraOpNumThreads(ort_session_options,
                                             session_options->intra_threads);
  ORT_CHECK_STATUS(
      status, "Failed to set intra op threads",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  status = g_state.api->SetInterOpNumThreads(ort_session_options,
                                             session_options->inter_threads);
  ORT_CHECK_STATUS(
      status, "Failed to set inter op threads",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Optimization settings
  status = g_state.api->SetSessionGraphOptimizationLevel(
      ort_session_options, session_options->graph_optimization_level);
  ORT_CHECK_STATUS(
      status, "Failed to set graph optimization level",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Make dynamic axes static if specified
  if (session_options->dynamic_axes) {
    set_dynamic_dimensions(ort_session_options, session_options->dynamic_axes);
  }

  // Disable thread spinning
  status = g_state.api->AddSessionConfigEntry(
      ort_session_options, "session.intra_op.allow_spinning", "0");
  ORT_CHECK_STATUS(
      status, "Failed to disable thread spinning",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Create session with appropriate provider
  OrtSession* ort_session = NULL;

  // CPU provider (default)
  void* custom_cpu_op_library_handle = NULL;

  status = g_state.api->RegisterCustomOpsLibrary(
      ort_session_options, ort_extensions_lib_path(), &custom_cpu_op_library_handle);
  ORT_CHECK_STATUS(
      status, "Failed to register custom ops library",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Create the session
  status = g_state.api->CreateSession(g_state.env, onnx_model_path,
                                      ort_session_options, &ort_session);
  ORT_CHECK_STATUS(
      status, "Failed to create ONNX Runtime session",
      g_state.api->ReleaseSessionOptions(ort_session_options), NULL);

  // Clean up session options
  g_state.api->ReleaseSessionOptions(ort_session_options);
  if (status) g_state.api->ReleaseStatus(status);

  return ort_session;
}

// Free ORT Session
void freeORTSession(OrtSession* ort_session) {
  if (ort_session) g_state.api->ReleaseSession(ort_session);
}

static void freeMemInterface(OrtMemoryInfo* memory_info, OrtIoBinding* io_binding) {
  if (memory_info) g_state.api->ReleaseMemoryInfo(memory_info);
  if (io_binding) g_state.api->ReleaseIoBinding(io_binding);
}

// Inference using ONNX Runtime with IO Binding
static OrtStatus* InferenceWithIOBinding(OrtSession* session, OrtValue** inputs, size_t input_count,
                                         const char* memory_device, OrtValue*** outputs,
                                         size_t* output_count_ptr, int64_t* inference_time_ns) {
  OrtStatus* status = NULL;               // For storing error status
  OrtIoBinding* io_binding = NULL;        // IO binding object
  OrtMemoryInfo* memory_info = NULL;      // Memory info for target device
  OrtAllocator* allocator = NULL;         // Default allocator
  struct timespec start_time, end_time;   // For timing measurements
  *inference_time_ns = 0;                 // Inference time is 0 until inference completes

  /* Validate input parameters */
  if (!session || !inputs || !outputs || !output_count_ptr || !inference_time_ns) {
    status = g_state.api->CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments");
    return status;
  }

  /* Step 0: Run inference with timing measurement */
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  /* Step 1: Create IO binding object */
  status = g_state.api->CreateIoBinding(session, &io_binding);
  if (status != NULL) return status;

  /* Step 2: Create memory info for the target device */
  if (strcmp(memory_device, "cpu") == 0) {
    status = g_state.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  } else {
    status = g_state.api->CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported memory device");
    freeMemInterface(NULL, io_binding);
    return status;
  }

  /* Step 3: Get the default allocator */
  status = g_state.api->GetAllocatorWithDefaultOptions(&allocator);
  if (status != NULL) return status;

  /* Step 4: Bind all input tensors to the IO binding */
  for (size_t i = 0; i < input_count; i++) {
    // Get the name of the current input
    char* input_name = NULL;
    status =
        g_state.api->SessionGetInputName(session, i, allocator, &input_name);
    if (status != NULL) {
      freeMemInterface(memory_info, io_binding);
      return status;
    }

    // Bind the input tensor to its name in the model
    status = g_state.api->BindInput(io_binding, input_name, inputs[i]);
    if (status != NULL) {
      freeMemInterface(memory_info, io_binding);
      return status;
    }

    // Free the input name string
    status = g_state.api->AllocatorFree(allocator, input_name);
    if (status != NULL) {
      freeMemInterface(memory_info, io_binding);
      return status;
    }
  }

  /* Step 5: Get output count and bind all outputs */
  status = g_state.api->SessionGetOutputCount(session, output_count_ptr);
  if (status != NULL) {
    freeMemInterface(memory_info, io_binding);
    return status;
  }

  /* Step 6: Bind all output tensors to the IO binding */
  for (size_t i = 0; i < *output_count_ptr; i++) {
    // Get the name of the current output
    char* output_name = NULL;
    status = g_state.api->SessionGetOutputName(session, i, allocator, &output_name);
    if (status != NULL) {
      freeMemInterface(memory_info, io_binding);
      return status;
    }

    // Bind this output to the target device
    status = g_state.api->BindOutputToDevice(io_binding, output_name, memory_info);
    if (status != NULL) {
      freeMemInterface(memory_info, io_binding);
      return status;
    }

    status = g_state.api->AllocatorFree(allocator, output_name);
    if (status != NULL) {
      freeMemInterface(memory_info, io_binding);
      return status;
    }
  }

  /* Step 7: Execute the model with the bound inputs/outputs */
  status = g_state.api->RunWithBinding(session, NULL, io_binding);
  if (status != NULL) {
    freeMemInterface(memory_info, io_binding);
    return status;
  }

  /* Step 8: Retrieve the output tensor values */
  size_t realized_output_count = 0;
  status = g_state.api->GetBoundOutputValues(io_binding, allocator, outputs, &realized_output_count);
  if (status != NULL) {
    freeMemInterface(memory_info, io_binding);
    return status;
  }
  ASSERT_TRUE(realized_output_count == *output_count_ptr);

  clock_gettime(CLOCK_MONOTONIC, &end_time);

  // Calculate inference time in nanoseconds
  *inference_time_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000L + (end_time.tv_nsec - start_time.tv_nsec);

  /* Step 9: Clean up resources */
  freeMemInterface(memory_info, io_binding);
  if (status) g_state.api->ReleaseStatus(status);

  return NULL;
}

// Run inference on a model with input tensors
bool ORT_RunModel(const char* model_path, OrtValue** input_tensors,
                  size_t input_count, OrtResult* result,
                  const ORTSessionOptions* session_options) {
  OrtStatus* status = NULL;

  if (!model_path || !input_tensors || !input_count || !result) {
    status = g_state.api->CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments");
    ORT_CHECK_STATUS(status, "ORT_RunModel failed", NULL, false);
  }

  // Create ORT Session
  OrtSession* session = initInference(model_path, session_options);
  if (!session) {
    status = g_state.api->CreateStatus(ORT_RUNTIME_EXCEPTION, "ORT Session cannot be initialized");
    ORT_CHECK_STATUS(status, "ORT_RunModel failed", NULL, false);
  }

  // Run inference using Inference
  status = InferenceWithIOBinding(session, input_tensors, input_count, session_options->memory_device,
    &result->outputs, &result->output_count, &result->inference_time_ns);
  if (status != NULL) {
    ORT_CHECK_STATUS(status, "ORT_RunModel failed", freeORTSession(session), false);
  }

  // Release session
  freeORTSession(session);
  if (status) g_state.api->ReleaseStatus(status);

  return true;
}

// Free OrtResult structure
void freeORTResult(OrtResult* result) {
  if (!result) return;

  OrtStatus* status = NULL;

  // Release all output tensors
  if (result->outputs) {
    for (size_t i = 0; i < result->output_count; i++) {
      if (result->outputs[i]) g_state.api->ReleaseValue(result->outputs[i]);
    }

    // Free the outputs array
    if (g_state.allocator) {
      status = g_state.api->AllocatorFree(g_state.allocator, result->outputs);
      if (status != NULL) ORT_CHECK_STATUS_VOID(status, "freeORTResult failed", free(result->outputs));
    }

    else free(result->outputs);
  }

  free(result);
}
