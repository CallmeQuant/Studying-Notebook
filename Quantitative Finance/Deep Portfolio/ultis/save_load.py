import os
import torch
import torch.nn as nn

def _retrieve_model_ct(model_path, model_name):
    """Get the model file that stores network parameters"""
    return os.path.join(model_path, f"best_model_weight_{model_name}.pt")

def _retrieve_model_config(model_path, model_name):
    """Get the model file that stores configurations/hyperparameters"""
    return os.path.join(model_path, f"best_model_config_{model_name}.json")

def save_model(model, model_path, model_name = 'gru'):
    os.makedirs(model_path, exist_ok=True)
    model_file = _retrieve_model_ct(model_path, model_name)
    if os.path.isfile(model_file):
      print("Model checkpoint is already existed. Will be overwritten")
    checkpoint = {}
    if isinstance(model, nn.DataParallel):
        checkpoint["model_state_dict"] = model.module.state_dict()
    else:
        checkpoint["model_state_dict"] = model.state_dict()
    torch.save(checkpoint, model_path + f"/best_model_weight_{model_name}.pt")

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(
    model,
    model_path,
    model_name,
    device = None,
    eval=True):
    if device is None:
      device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    model_file =  _retrieve_model_ct(model_path, model_name)
    net = model.to(device)
    checkpoint = torch.load(model_file, map_location=device)
    if isinstance(net, nn.DataParallel):
        net.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        new_dict = remove_prefix(checkpoint["model_state_dict"], "module.")
        net.load_state_dict(new_dict, strict=False)
    if eval:
        net.eval()
    return net