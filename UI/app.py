import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import torchvision.transforms as transforms


class Block(nn.Module):
    def __init__(self,in_channels,out_channels,down=True,act='relu',use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,2,1,bias=False,padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(in_channels,out_channels,4,2,1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act =='relu' else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class Generator (nn.Module):
        def __init__(self,in_channels = 3,features =64):
            super().__init__()
            self.initial_down = nn.Sequential(
                nn.Conv2d(in_channels,features,4,2,1,padding_mode='reflect'),
                nn.LeakyReLU(0.2),
            )#128

            self.down1 = Block(features,features*2,down=True,act='leaky',use_dropout=False)#64
            self.down2 = Block(features*2,features*4,down=True,act='leaky',use_dropout=False)#32
            self.down3 = Block(features*4,features*8,down=True,act='leaky',use_dropout=False)#16
            self.down4 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)#8
            self.down5 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)#4
            self.down6 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)#2
            self.down7 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)#2
            self.bottleneck = nn.Sequential(
                nn.Conv2d(features*8,features*8,4,2,1,padding_mode = 'reflect'),nn.ReLU()  #1x1
            )
            self.up1 = Block(features*8,features*8,down=False,act='relu',use_dropout=True)
            self.up2 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
            self.up3 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
            self.up4 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=False)
            self.up5 = Block(features*8*2,features*4,down=False,act='relu',use_dropout=False)
            self.up6 = Block(features*4*2,features*2,down=False,act='relu',use_dropout=False)
            self.up7 = Block(features*2*2,features,down=False,act='relu',use_dropout=False)
            self.final_up = nn.Sequential(
                nn.ConvTranspose2d(features*2,in_channels,kernel_size=4,stride=2,padding=1),
                nn.Tanh() #pixel value between -1 and 1 
            )


        def forward (self,x):
            d1 = self.initial_down(x)
            d2 = self.down1(d1)
            d3 = self.down2(d2)
            d4 = self.down3(d3)
            d5 = self.down4(d4)
            d6 = self.down5(d5)
            d7 = self.down6(d6)
            bottleneck=self.bottleneck(d7)
            up1 = self.up1(bottleneck)
            up2 = self.up2(torch.cat([up1,d7],1))
            up3 = self.up3(torch.cat([up2,d6],1))
            up4 = self.up4(torch.cat([up3,d5],1))
            up5 = self.up5(torch.cat([up4,d4],1))
            up6 = self.up6(torch.cat([up5,d3],1))
            up7 = self.up7(torch.cat([up6,d2],1))
            return self.final_up(torch.cat([up7,d1],1))
        

# Set up page config
st.set_page_config(page_title="Sketch to Real Image Converter", page_icon="🎨")

# Title and description
st.title("🎨 Sketch to Real Image Converter")
st.write("Upload your sketch and for conversion! ")

# Load the saved generator model
@st.cache_resource
def load_generator(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize generator with matching architecture
    generator = Generator(in_channels=3, features=64)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Load weights into model
    generator.load_state_dict(state_dict)
    
    generator.eval()
    return generator.to(device)

# Image preprocessing (adjusted for your model)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB images
    ])
    return transform(image).unsqueeze(0)

# Post-process generated image (adjusted for tanh activation)
def postprocess_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor * 0.5 + 0.5  # Scale from [-1,1] to [0,1]
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return image

with st.sidebar:
    st.header("Model Configuration")
    model_path = st.text_input("Generator model path", "final_generator.pth")

# Main interface
uploaded_file = st.file_uploader("Upload your sketch", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Sketch")
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, use_column_width=True)

    # Generate real image
    if st.button("Generate Real Image"):
        try:
            # Load model
            generator = load_generator(model_path= "final_generator.pth")
            
            # Preprocess image
            input_tensor = preprocess_image(uploaded_image)
            
            # Generate image
            with torch.no_grad():
                generated_tensor = generator(input_tensor)
            
            # Post-process and display
            generated_image = postprocess_image(generated_tensor)
            
            with col2:
                st.subheader("Generated Image")
                st.image(generated_image, use_column_width=True)
                
                # Create download button
                buffered = io.BytesIO()
                generated_image.save(buffered, format="JPEG", quality=95)
                st.download_button(
                    label="Download Generated Image",
                    data=buffered.getvalue(),
                    file_name="generated_image.jpg",
                    mime="image/jpeg"
                )
        
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")