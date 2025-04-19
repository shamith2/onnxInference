/*
 * File: test_onnxInference.c
 * Description: Test cases for onnxInference.c
 * Author: Shamith Achanta
 */

#include "onnxInference.h"

void test_null_data() {
    int64_t shape[1] = {2};
    OrtValue* tensor = createTensor(NULL, shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_TRUE(tensor == NULL);
}

void test_null_shape() {
    float data[2] = {1.0f, 2.0f};
    OrtValue* tensor = createTensor(data, NULL, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_TRUE(tensor == NULL);
}

void test_create_float_tensor() {
    float data[6] = {1.0f, 11.0f, 21.0f, 42.0f, 56.0f, 65.0f};
    int64_t shape[2] = {2, 3};
  
    OrtValue* tensor = createTensor(data, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_TRUE(tensor != NULL);

    inspectTensor(tensor, true, 6);
  
    if (tensor) freeTensor(tensor);
}

void test_create_string_tensor() {
    const char* strings[] = {"hello", "world"};
    int64_t shape[1] = {2};

    OrtValue* tensor = createTensor(strings, shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    ASSERT_TRUE(tensor != NULL);

    inspectTensor(tensor, true, 2);

    if (tensor) freeTensor(tensor);
}

void test_create_batch_string_tensor() {
    const char* strings[4][3] = {{"\0", "hello", "world"}, {"how", "are", "you"}, {"here", "i", "am"}, {"catch", "me", "\0"}};
    int64_t shape[2] = {4, 3};

    OrtValue* tensor = createTensor(strings, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    ASSERT_TRUE(tensor != NULL);

    inspectTensor(tensor, true, 12);

    if (tensor) freeTensor(tensor);
}

void test_invalid_shape_value() {
    float data[2] = {1.0f, 2.0f};
    int64_t shape[1] = {-1};  // Invalid negative dimension
    OrtValue* tensor = createTensor(data, shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_TRUE(tensor == NULL);
}

void test_invalid_type() {
    float data[2] = {1.0f, 2.0f};
    int64_t shape[1] = {2};
    OrtValue* tensor = createTensor(data, shape, 1, (ONNXTensorElementDataType) 99);
    ASSERT_TRUE(tensor == NULL);
}

void test_init_session_options() {
    ORTSessionOptions* options = initSessionOptions("CPU", "cpu", NULL, 1, 2, ORT_ENABLE_ALL, ORT_SEQUENTIAL, ORT_EXECUTION_ORDER_DEFAULT, ORT_LOGGING_LEVEL_INFO);
    ASSERT_TRUE(options != NULL);

    ASSERT_TRUE(strcmp(options->execution_provider, "CPU") == 0);
    ASSERT_TRUE(strcmp(options->memory_device, "cpu") == 0);
    ASSERT_TRUE(options->dynamic_axes == NULL);
    ASSERT_TRUE(options->intra_threads == 1);
    ASSERT_TRUE(options->inter_threads == 2);
    ASSERT_TRUE(options->graph_optimization_level == ORT_ENABLE_ALL);
    ASSERT_TRUE(options->execution_mode == ORT_SEQUENTIAL);
    ASSERT_TRUE(options->execution_order == ORT_EXECUTION_ORDER_DEFAULT);
    ASSERT_TRUE(options->log_severity_level == ORT_LOGGING_LEVEL_INFO);

    if (options) freeSessionOptions(options);
}

void test_simple_gemm() {
    ORTSessionOptions* options = initSessionOptions("CPU", "cpu", NULL, 1, 2, ORT_ENABLE_ALL, ORT_SEQUENTIAL, ORT_EXECUTION_ORDER_DEFAULT, ORT_LOGGING_LEVEL_INFO);
    ASSERT_TRUE(options != NULL);

    const int64_t shape[2] = {2, 256};
    int low = 100;
    int high = 1000;
    float data[512];

    for (int i = 0; i < 512; i++) {
        data[i] = (float) (rand() % (high - low + 1) + low);
    }

    OrtValue* input = createTensor(data, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_TRUE(input != NULL);
    OrtResult* result = malloc(1 * sizeof(OrtResult*));
    ASSERT_TRUE(result != NULL);

    bool retval = ORT_RunModel("simple_gemm.onnx", &input, 1, result, options);
    ASSERT_TRUE(retval == true);
    ASSERT_TRUE(result != NULL);
    ASSERT_TRUE(result->output_count == 1);

    if (retval) inspectTensor(result->outputs[0], false, 0);

    if (input) freeTensor(input);
    if (result) freeORTResult(result);
    if (options) freeSessionOptions(options);
}

int basic_tests() {
    ASSERT_TRUE(ORT_Initialize());

    test_null_data();
    test_null_shape();
    test_create_float_tensor();
    test_create_string_tensor();
    test_create_batch_string_tensor();
    test_invalid_shape_value();
    test_invalid_type();
    test_init_session_options();
    test_simple_gemm();

    return 0;
}
