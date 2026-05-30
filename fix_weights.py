import os
import shutil
from safetensors.torch import load_file, save_file

lora_dir = r"D:\multi_copy\checkpoint-408"
file_path = os.path.join(lora_dir, "adapter_model.safetensors")
backup_path = os.path.join(lora_dir, "adapter_model_backup.safetensors")
temp_path = os.path.join(lora_dir, "temp.safetensors")

if not os.path.exists(backup_path):
    shutil.copy2(file_path, backup_path)

tensors = load_file(file_path)

new_tensors = {}
for key, value in tensors.items():
    new_key = key.replace("model.model.language_model", "model.language_model")
    new_tensors[new_key] = value.clone()

save_file(new_tensors, temp_path)

del tensors
del new_tensors

os.remove(file_path)
os.rename(temp_path, file_path)

print("Fix completed")