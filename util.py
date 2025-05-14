import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from io import BytesIO
import base64
import pickle

__metadata = None

def tensor_to_color_base64(output_tensor,top_i):
    vmin = output_tensor.min().detach()
    vmax = output_tensor.max().detach()
    processed_image = torch.clip(output_tensor,min=vmin,max=vmax).squeeze(0).detach().permute(1,2,0)[:,:,top_i]
    # Normalize to [0,1]
    processed_image = (processed_image - vmin) / (vmax - vmin + 1e-8)
    imgg = processed_image
    cmap = plt.get_cmap("viridis")
    
    cmap_var = cmap(imgg)
    colored_8bit = (cmap_var * 255).astype(np.uint8)
    img = Image.fromarray(colored_8bit)
    # save to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
def deconvolution(activation,conv_layer):
    deconv = nn.ConvTranspose2d(
        in_channels=conv_layer["out_channels"],
        out_channels=conv_layer["in_channels"],
        kernel_size=conv_layer["kernel_size"],
        stride=conv_layer["stride"],
        padding=conv_layer["padding"],
        bias=False
    )
    deconv.weight.data = torch.tensor(conv_layer["weights"])
    return deconv(activation)

def get_layer_tensor(index:str):
    ## get file conolution file path
    layer_file_path = __metadata["layers"][index]["file"]
    module_tensor = None
    try:
        with open(f"./{layer_file_path}",'rb') as f:
            module_tensor = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the required files: {e}")
    ## Output from the convolution
    output_tensor = torch.tensor(module_tensor["output"])
    ## Calculate the approximate input
    approximate_input = deconvolution(output_tensor,module_tensor["module"])
    ## The Top approximate input, I also want to show case the top 9 active inputs
    top_values, top_indices = torch.topk(torch.mean(approximate_input,dim=(2,3)), k=6)

    response = []
    for idx,top_i in enumerate(top_indices[0].numpy()):
        approx_input_decoded = tensor_to_color_base64(approximate_input,top_i)
        feature_map_decoded = tensor_to_color_base64(output_tensor,top_i)
        response.append({
            "index":idx+1,
            "approximate_input":approx_input_decoded,
            "approximate_output":feature_map_decoded,
        })
    return response

def get_all_layers():
    meta_data_info = __metadata["model_info"]
    total_layers = meta_data_info["total_layers"]
    indices = np.arange(total_layers)
    response = []
    for index in indices:
        info_layer = get_layer_tensor(index+1)
        response.append(info_layer)
    return response
    

def load_artifacts():
    global __metadata
    try:
        with open("./modeldatapickle/metadata.json", "r") as f:
            __metadata = json.load(f)
        # with open("./artifacts/category_indexs.pickle", "rb") as f:
        #     __category_to_index = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the required files: {e}")
    except Exception as e:
        print(f"Error loading models: {e}")

if __name__ == "__main__":
    load_artifacts()
    # print(__metadata)
    # print(get_layer_tensor(1))
    print(get_all_layers()[:2])