from safetensors.torch import load_file
import torch

safetensor_path = "D:/aiml_partb_3/report_predictor/maintenance/saved_model/model.safetensors"
pytorch_model_path = "D:/aiml_partb_3/report_predictor/maintenance/saved_model/pytorch_model.bin"

# Load safetensors model
model_state_dict = load_file(safetensor_path)

# Save it as pytorch_model.bin
torch.save(model_state_dict, pytorch_model_path)

print("Conversion complete: model.safetensors → pytorch_model.bin")
