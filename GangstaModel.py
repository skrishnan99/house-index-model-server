import sys
sys.path.append('Tag2Text')

import os
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as TS
from Tag2Text.models import tag2text
from PIL import Image
from Tag2Text import inference_ram

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def collate_data(mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = np.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    json_data = {
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    
    return json_data, mask_img

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

class GangstaModel:
    def __init__(
        self, 
        config_file="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ram_checkpoint="./Tag2Text/pretrained/ram_swin_large_14m.pth", 
        grounded_checkpoint="groundingdino_swint_ogc.pth", 
        sam_checkpoint="sam_vit_h_4b8939.pth",
        box_threshold=0.25,
        text_threshold=0.2,
        iou_threshold=0.5,
        device="cuda"
    ):

        # initialize Recognize Anything Model
        self.ram_model = tag2text.ram(pretrained=ram_checkpoint,
                                            image_size=384,
                                            vit='swin_l')
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(device)
        
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.transform = TS.Compose([
                        TS.Resize((384, 384)),
                        TS.ToTensor(), normalize
                    ])


        self.grounded_dino_model = load_model(config_file, grounded_checkpoint, device=device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold

        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

        self.device = device
    

    def predict_ram(self, pil_image):

        raw_image = pil_image.resize(
                    (384, 384))
        raw_image = self.transform(raw_image).unsqueeze(0).to(self.device)

        res = inference_ram.inference(raw_image, self.ram_model)

        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        image_tags = res[0].replace(' |', ',')

        return image_tags
    
    def predict_grounded_dino(self, img_tensor, img_tags, orig_img_height, orig_img_width):
        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.grounded_dino_model, img_tensor, img_tags, self.box_threshold, self.text_threshold, device=self.device
        )

        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([orig_img_width, orig_img_height, orig_img_width, orig_img_height])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")

        return boxes_filt, pred_phrases
    

    def run_sam_inference(self, rgb_img, boxes_filt, orig_img_height, orig_img_width):

        self.sam_predictor.set_image(rgb_img)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, (orig_img_height, orig_img_width)).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        return masks


    def run_gangsta_inference(self, img):

        image_pil = Image.fromarray(img)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)


        size = image_pil.size
        orig_img_height, orig_img_width = size[1], size[0]

        image_tags = self.predict_ram(pil_image=image_pil)

        boxes_filt, pred_phrases = self.predict_grounded_dino(img_tensor=image_tensor, img_tags=image_tags, orig_img_height=orig_img_height, orig_img_width=orig_img_width)

        masks = self.run_sam_inference(rgb_img=img, boxes_filt=boxes_filt, orig_img_height=orig_img_height, orig_img_width=orig_img_width)

        json_data, mask_img = collate_data(masks, boxes_filt, pred_phrases)

        return json_data, mask_img

if __name__ == "__main__":
    image_path="assets/rand_cam_test.jpg"
    img = cv2.imread(image_path)[:,:,::-1]

    init_start = time.perf_counter()
    gangsta_model = GangstaModel()
    print("time to initialize the gangsta model class", time.perf_counter() - init_start)

    infer_start = time.perf_counter()
    json_data, mask_img = gangsta_model.run_gangsta_inference(img=img)
    cv2.imwrite("mask.png", mask_img)
    print("infer time", time.perf_counter() - infer_start)