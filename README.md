# onnxInference: Inference with ONNX Models for AI Applications
Getting to understand efficient inference with ONNX models for practical applications and pipelines

- onnxHelpers/onnxBenchmark.py = script to convert pytorch model to onnx, quantize fp32 onnx models to int8, and run benchmark inference on AMD Ryzen AI processor

### Custom AI Recall Pipeline:
 * Implemented custom AI Recall feature, similar to Microsoft Windows AI Recall feature, running locally with Phi-3 Vision model for describing/analysing screenshots and Phi-3 Mini model to rename the screenshots based on the image description geneated by the vision model.
 
 * The filenames and descriptions (after chunking) are stored in a simple database for Retrieval-Augmented Generation (RAG). Based on a query, given by the user, the descriptions, along with the associated filenames of the screenshots, that are similar to the query are retrieved. The Phi-3 models have been tested on the CPU

 * Example Run 1:
   
   ![airecall_1](https://github.com/shamith2/onnxInference/assets/43729418/89e3fccf-5747-479c-992d-451e9332bc51)

 * Best Result:
   
   ![best_result](https://github.com/shamith2/onnxInference/blob/1f1b3a81d6dfbbd2cab0321e70bbaf3f10790970/results/aiRecall/snapshots/20240610/YouTube_Keynote_CopilotPC_SatyaNadella_Subscribe_6102024_164832.png)

 * Once, the descriptions are added into the database, subsequent retrivals are quick (test screenshots and database saved in results/aiRecall/snapshots; these screenshots are not very diverse)
   
   ![ai_recall_2](https://github.com/shamith2/onnxInference/assets/43729418/875011d7-43ca-4a2f-b7b2-d828884409a0)


### Stable Diffusion Pipeline:
  * Scripts to run stable diffusion pipeline, currently on running on DirectML-supported devices

  ![image](https://github.com/shamith2/onnxInference/blob/db91c3483d4ad8f8ab8d5dc2a1379b03268bebb3/results/stableDiffusion/sd_turbo_results/SD%202.1%20Turbo_visualize_1.png)
