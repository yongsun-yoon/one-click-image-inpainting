import os
import cv2
import torch
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image
from streamlit.runtime.scriptrunner import get_script_run_ctx

from streamlit_image_coordinates import streamlit_image_coordinates

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
from transformers import AutoProcessor, AutoModelForUniversalSegmentation


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()


def get_inpaint_config(**kwargs):
    data = dict(
        ldm_steps=1,
        ldm_sampler=LDMSampler.plms,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=200,
        hd_strategy_resize_limit=200,
    )
    data.update(**kwargs)
    return Config(**data)


@st.experimental_singleton
def load_models(device='cpu'):
    segment_processor = AutoProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
    segment_model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
    _ = segment_model.eval().requires_grad_(False).to(device)

    inpaint_config = get_inpaint_config()
    inpaint_model = ModelManager('lama', device=device)
    return segment_processor, segment_model, inpaint_config, inpaint_model


def run_segment(pil_image):
    segment_inputs = segment_processor(images=pil_image, task_inputs=["panoptic"], return_tensors="pt").to(device)
    segment_outputs = segment_model(**segment_inputs)
    segment_outputs = segment_processor.post_process_panoptic_segmentation(segment_outputs, target_sizes=[pil_image.size[::-1]])[0]
    segmentation = segment_outputs['segmentation']
    return segmentation

def run_inpaint(np_image, mask):
    dilated_mask = cv2.dilate(mask, None, iterations=3) 
    outputs = inpaint_model(np_image, dilated_mask, inpaint_config)
    outputs = outputs[:,:,[2,1,0]].astype(np.uint8)
    return outputs


def main():
    st.title('One-click Image Inpainting')

    session_id = get_script_run_ctx().session_id
    session_dir = f'results/{session_id}'
    os.makedirs(session_dir, exist_ok=True)

    original_path = f'{session_dir}/original.png'
    inpainted_path = f'{session_dir}/inpainted.png'

    img_file = st.file_uploader(label='Upload image', type=['png', 'jpg', 'jpeg'])
    if img_file is not None:
        pil_image = Image.open(img_file).convert('RGB')
        pil_image.save(original_path)
        segmentation = run_segment(pil_image)
    
    if os.path.exists(original_path) or os.path.exists(inpainted_path):
        st.write('Input')
        if os.path.exists(inpainted_path):
            pil_image = Image.open(inpainted_path)
        else:
            pil_image = Image.open(original_path)
        np_image = np.array(pil_image)
        value = streamlit_image_coordinates(pil_image)
            
        if value is not None:
            instance_id = segmentation[value['y'], value['x']].item()
            mask = torch.where(segmentation==instance_id, 1, 0)
            mask = mask.cpu().numpy() * 255.
            
            inpainted_image = run_inpaint(np_image, mask)
            inpainted_image = Image.fromarray(inpainted_image)
            inpainted_image.save(inpainted_path)
            
            st.write('Output')
            st.image(inpainted_image)

            buf = BytesIO()
            inpainted_image.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            btn = st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="image.png",
                mime="image/jpeg",
            )



if __name__ == '__main__':
    device = args.device
    segment_processor, segment_model, inpaint_config, inpaint_model = load_models(device=device)
    main()