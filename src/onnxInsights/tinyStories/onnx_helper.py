import gc
import onnx
import torch

def export_onnx(
        model: torch.nn.Module,
        model_name: str,
        onnx_model_path: str,
        input_names: list,
        model_inputs: list,
        use_external_data: bool = False
    ):
    if not isinstance(model, torch.nn.Module):
            raise Exception("Model has to be of type torch.nn.Module")
    
    with torch.no_grad():
        # set model to eval
        model.eval()
        
        print("\nConverting {} from PyTorch to ONNX...\n".format(model_name.capitalize()))
        
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.automatic_dynamic_shapes = True

        kwargs = {}

        for i in range(len(input_names)):
            kwargs[str(input_names[i])] = model_inputs[i]

        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
                    
        export_output = torch.onnx.dynamo_export(model, **kwargs, export_options=export_options)
        export_output.save(onnx_model_path)

    if use_external_data:
        print("\nSaving external data to one file...\n")

        # try freeing memory
        gc.collect()

        onnx_model = onnx.load(onnx_model_path, load_external_data=True)
        
        onnx.save_model(
            onnx_model,
            onnx_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=model_name + "-data",
            size_threshold=1024,
        )
    
    try:
        onnx.checker.check_model(onnx_model_path)

    except onnx.checker.ValidationError as e:
        raise Exception(e)

    print("\nSuccessfully converted PyTorch model to ONNX!!\n")

    return 0
