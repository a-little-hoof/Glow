import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision
from glow_grid import *

    
if __name__ == "__main__":
    # Load the image
    image_path = './10140.jpg'
    image = Image.open(image_path)

    # Define the transformation: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        #  transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
    ])

    device = "cuda"

    # Apply the transformation to the image
    image_tensor = transform(image)

    # Add a batch dimension (batch_size, channels, height, width)
    image_tensor = image_tensor.unsqueeze(0)

    print(image_tensor.shape)  # Should print: torch.Size([1, 3, 64, 64])

    image_tensor = image_tensor.to(device)
    model =  GlowGrid(input_shape=[3,256,256], num_flow=32, num_block=4).to(device)

    z, _,  jac = model(image_tensor)
    print(z.shape)
    out, jac2, log_p = model.reverse(z)

    # tensor = torch.
    # flow = CondFlow(in_channel=3, num_cond_layers=3).to(device)

    # output, logdet1 = flow(image_tensor, image_tensor)
    # input, logdet2  = flow.reverse(output, image_tensor, jac=True)

    print(jac+jac2)
    torchvision.utils.save_image(out, 'output_image.png')

    out, det, log_p, z_new = model.blocks[0](image_tensor)
    print(out.shape)
    image, logdet, log_p = model.blocks[0].reverse(out, z_new, jac=True)
    