import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

class DiffusionSimulator:
    def __init__(self, T=200):
        # 1. Math Setup
        self.T = T
        self.beta = torch.linspace(0.0001, 0.02, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        
        # State variables to keep the slider smooth
        self.fixed_noise = None
        
        # Image conversion pipelines
        self.to_tensor = transforms.Compose([
            transforms.Resize((256, 256)), # Standardize to keep processing instant
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def to_pil(self, tensor):
        """Helper to convert [-1, 1] tensor back to a viewable PIL Image"""
        tensor = (tensor.squeeze(0).clamp(-1, 1) + 1.0) / 2.0
        tensor = (tensor * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(tensor)

    def apply_noise(self, input_image, t):
        """The core Equation 25.3 implementation"""
        if input_image is None:
            return None
            
        # Convert uploaded PIL image to PyTorch tensor
        x_0 = self.to_tensor(input_image).unsqueeze(0)
        
        # Generate new fixed noise ONLY if it's a new image upload
        if self.fixed_noise is None or self.fixed_noise.shape != x_0.shape:
            self.fixed_noise = torch.randn_like(x_0)
            
        if t == 0:
            return self.to_pil(x_0)
            
        # Grab constants for the exact timestep 't'
        t_tensor = torch.tensor([t])
        sqrt_a_bar_t = self.sqrt_alpha_bar[t_tensor].view(-1, 1, 1, 1)
        sqrt_one_minus_a_bar_t = self.sqrt_one_minus_alpha_bar[t_tensor].view(-1, 1, 1, 1)
        
        # THE MASTER FORWARD EQUATION
        x_t = (sqrt_a_bar_t * x_0) + (sqrt_one_minus_a_bar_t * self.fixed_noise)
        
        return self.to_pil(x_t)

# Instantiate our simulator
sim = DiffusionSimulator(T=200)

# Build the Gradio UI
demo = gr.Interface(
    fn=sim.apply_noise,
    inputs=[
        gr.Image(type="pil", label="Upload Clean Image (x_0)"),
        gr.Slider(minimum=0, maximum=199, step=1, value=0, label="Timestep (t)")
    ],
    outputs=gr.Image(label="Noisy Image (x_t)"),
    title="Forward Diffusion Explorer",
    description="""Upload an image and slowly drag the slider. 
    Notice how the high-frequency details (textures) are destroyed first, 
    and leaves the low-frequency details (broad shapes) until the very end.""",
    live=True
)

if __name__ == "__main__":
    demo.launch()