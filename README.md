# Insights into ONNX Models for AI Applications
Getting to know more about onnx models: insights into their performance and bottlenecks for practical applications and pipelines

- onnxHelpers/onnxBenchmark.py = script to convert pytorch model to onnx, quantize fp32 onnx models to int8, and run benchmark inference on AMD Ryzen AI processor
- onnxHelpers/onnxTransformer.py = script to modify operators in onnx models

- stable-diffusion = scripts to run stable diffusion pipeline, currently on running on CPU + DirectML-supported devices

![image](https://github.com/shamith2/onnxInsights/blob/db91c3483d4ad8f8ab8d5dc2a1379b03268bebb3/results/stableDiffusion/sd_turbo_results/SD%202.1%20Turbo_visualize_1.png)

For running and testing onnx models, I am currently running them on AMD Ryzen APUs using Radeon iGPU + Ryzen AI processor
refer to https://ryzenai.docs.amd.com/en/latest/inst.html for installation and config files
