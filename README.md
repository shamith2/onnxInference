# onnx InsightAI
Getting to know more about onnx models, insights into their performance and bottlenecks for practical applications and pipelines

onnxHelpers/onnxBenchmark.py = script to convert pytorch model to onnx, quantize fp32 onnx models to int8, and run benchmark inference on AMD Ryzen AI processor
onnxHelpers/onnxTransformer.py = script to modify operators in onnx models

stable-diffusion = scripts to run stable diffusion pipeline, currently on DirectML-supported devices

![image](https://github.com/shamith2/ryzenAI/assets/43729418/086c4869-51d3-4b1e-8473-165cadf29647)

For running and testing onnx models, I am currently running them on AMD Ryzen APUs using Radeon iGPU + Ryzen AI processor
refer to https://ryzenai.docs.amd.com/en/latest/inst.html for installation and config files
