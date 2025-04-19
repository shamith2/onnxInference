/*
 * File: onnxInference.h
 * Description:
 *    Header file defining helper functions for ONNX model inference using ONNX Runtime
 *
 * Author: Shamith Achanta
 */

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "onnxruntime_c_api.h"

// Struct declarations
typedef struct OrtSession OrtSession;
typedef struct OrtValue OrtValue;
typedef struct OrtTypeInfo OrtTypeInfo;
typedef struct OrtTensorTypeAndShapeInfo OrtTensorTypeAndShapeInfo;
typedef struct OrtEnv OrtEnv;
typedef struct OrtStatus OrtStatus;

// API Version
#define ORT_WRAPPER_API_VERSION_MAJOR 0
#define ORT_WRAPPER_API_VERSION_MINOR 1

// hard-coded for now
#define ORT_EXTENSIONS_VERSION "0.14.0"

// Global state
typedef struct {
  const OrtApi* api;
  OrtEnv* env;
  bool initialized;
  OrtAllocator* allocator;
} OrtGlobalState;

// Structure to hold tensor information
typedef struct {
  ONNXTensorElementDataType dtype;
  int64_t* dimensions;
  size_t ndims;
  size_t nelements;
} TensorInfo;

// Structure to hold model information
typedef struct {
  TensorInfo* inputs;
  size_t input_count;
  TensorInfo* outputs;
  size_t output_count;
} ModelInfo;

// Execution order options
typedef enum {
  ORT_EXECUTION_ORDER_DEFAULT = 0,
  ORT_EXECUTION_ORDER_BREADTH_FIRST = 1,
  ORT_EXECUTION_ORDER_TOPOLOGICAL = 2
} ExecutionOrder;

// Dynamic axis structure
typedef struct {
  const char* name;
  int64_t value;
} DynamicAxis;

typedef struct {
  DynamicAxis* dims;
  size_t count;
} DynamicAxes;

// Result structure to hold inference outputs
typedef struct {
  OrtValue** outputs;
  size_t output_count;
  int64_t inference_time_ns;
} OrtResult;

// Session options struct
typedef struct {
  const char* execution_provider;
  const char* memory_device;
  DynamicAxes* dynamic_axes;
  int intra_threads;
  int inter_threads;
  GraphOptimizationLevel graph_optimization_level;
  ExecutionMode execution_mode;
  ExecutionOrder execution_order;
  OrtLoggingLevel log_severity_level;
} ORTSessionOptions;

// Error checking helper with improved error handling
#define ORT_CHECK_STATUS(status, message, cleanup_fn, retval)                                        \
  do {                                                                                               \
    if (status != NULL) {                                                                            \
      LOG_MESSAGE(LOG_ERROR, "%s: %s\n", (message), g_state.api->GetErrorMessage(status));           \
      g_state.api->ReleaseStatus(status);                                                            \
      cleanup_fn;                                                                                    \
      return (retval);                                                                               \
    }                                                                                                \
  } while (0)

#define ORT_CHECK_STATUS_VOID(status, message, cleanup_fn)                                         \
do {                                                                                               \
  if (status != NULL) {                                                                            \
    LOG_MESSAGE(LOG_ERROR, "%s: %s\n", (message), g_state.api->GetErrorMessage(status));           \
    g_state.api->ReleaseStatus(status);                                                            \
    cleanup_fn;                                                                                    \
    return;                                                                                        \
  }                                                                                                \
} while (0)

// Enum for log levels
typedef enum { LOG_INFO, LOG_DEBUG, LOG_WARNING, LOG_ERROR } LogLevel;

// Macro to log messages with different log levels
#define LOG_MESSAGE(level, format, ...)                                     \
  do {                                                                      \
    const char* level_str;                                                  \
    switch (level) {                                                        \
      case LOG_INFO:    level_str = "INFO"; break;                          \
      case LOG_DEBUG:   level_str = "DEBUG"; break;                         \
      case LOG_WARNING: level_str = "WARNING"; break;                       \
      case LOG_ERROR:   level_str = "ERROR"; break;                         \
      default:          level_str = "INFO"; break;                          \
    }                                                                       \
    fprintf(stderr, "[%s] " format "\n", level_str, ##__VA_ARGS__);         \
  } while (0)

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define ASSERT_TRUE(x)                                                      \
  do {                                                                      \
    if (!(x)) {                                                             \
      fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, #x);        \
    }                                                                       \
  } while(0)

#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))

// Global OrtApi instance
extern OrtGlobalState g_state;

/* Helper Functions */
char* path_join(const char* path1, const char* path2);

/* API Functions */

// Function to create ORTSessionOptions
ORTSessionOptions* initSessionOptions(
  const char* execution_provider, const char* memory_device,
  DynamicAxes* dynamic_axes, int intra_threads, int inter_threads,
  int graph_optimization_level, int execution_mode, int execution_order, int log_severity_level);

// Function to free ORTSessionOptions
void freeSessionOptions(ORTSessionOptions* session_options);

// Initialize the ONNX Runtime API
bool ORT_Initialize(void);

// Check if ORT is initialized
bool ORT_IsInitialized(void);

// Clean up and release all resources
void ORT_Cleanup(void);

// Initialize ONNX Runtime inference session
OrtSession* initInference(const char* onnx_model_path, const ORTSessionOptions* session_options);

// Free ORT Session
void freeORTSession(OrtSession* ort_session);

// Get element size from data type
size_t GetElementSize(ONNXTensorElementDataType type);

// Get data type of Tensor
const char* get_tensor_element_type_string(ONNXTensorElementDataType type);

// Create an ONNX tensor with specified shape and data
OrtValue* createTensor(const void* data, const int64_t* shape,
                       size_t dim_count,
                       ONNXTensorElementDataType element_type);

// Release a tensor
void freeTensor(OrtValue* tensor);

// Run inference on a model with input tensors
bool ORT_RunModel(const char* model_path, OrtValue** input_tensors,
                  size_t input_count, OrtResult* result,
                  const ORTSessionOptions* session_options);

// Free OrtResult structure
void freeORTResult(OrtResult* result);

// Get Tensor Info
void inspectTensor(OrtValue* tensor, bool verbose, size_t num_elements);

// Get ONNX model inputs and outputs shape
void inspectModel(OrtSession* session);

// Free tensor info resources
void freeTensorInfo(TensorInfo* info);

// Free model info resources
void freeModelInfo(ModelInfo* info);

// Print tensor info to any file or string buffer
void printTensorInfo(TensorInfo* info, FILE* stream);

// Print model info to any file or string buffer
void printModelInfo(ModelInfo* info, FILE* stream);
