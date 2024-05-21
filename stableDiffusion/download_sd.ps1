optimum-cli export onnx --model stabilityai/stable-diffusion-2-1 --framework pt --cache_dir .cache sd-v2-onnx/

optimum-cli export onnx --model stabilityai/sdxl-turbo --framework pt --cache_dir .cache sdxl-turbo-onnx/

git clone https://huggingface.co/gfodor/sdxl-turbo-fp16-onnx

optimum-cli export onnx --model stabilityai/sd-turbo --framework pt --cache_dir .cache sd-turbo-onnx/