from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from segment_anything import SamPredictor,sam_model_registry
import time
from PIL import Image
import random
import torch
import yaml
import cv2
import numpy as np

path = 'models/yolov5m_Objects365.pt'
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
bert_model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
yolo_obj365 = torch.hub.load('ultralytics/yolov5', 'custom', path)
sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
sam.to('cuda')
predictor = SamPredictor(sam)
class contextAwareTool:
    def __init__(self):
        self.blip_model = blip_model
        self.bert_model = bert_model
        self.yolo = yolo_obj365
        self.obj365_labels = self._get_obj365_labels()

    def _get_similarity(self, text_1, text_2):
        sentence_embeddings = self.bert_model.encode([text_1, text_2])
        similarity_scores = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])
        return similarity_scores[0][0]

    def _get_img_caption(self, img_path):
        raw_image = Image.open(img_path).convert('RGB')
        text = "a picture of "
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
        out = self.blip_model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)

    def _get_detection_obj365(self, img_path):
        img = cv2.imread(img_path)
        res = self.yolo(img)
        return res

    def _get_obj365_labels(self):
        with open('Objects365.yaml', 'r', encoding='utf-8') as file:
            labels = yaml.load(file, Loader=yaml.FullLoader)
        obj365_labels = list(labels['names'].values())
        return obj365_labels

    def _get_most_relevent_object_bbox(self, base_img, obj_img):
        similarity_res = []
        obj_res = self._get_detection_obj365(obj_img).pandas().xyxy[0]
        base_img_caption = self._get_img_caption(base_img)
        for i in range(len(obj_res)):
            similarity_res.append([i, self._get_similarity(base_img_caption, obj_res.iloc[i, -1])])
        index = sorted(similarity_res, key=lambda x: x[1], reverse=True)[0][0]
        # print(index)
        coor = [int(x) for x in obj_res.iloc[index].tolist()[:4]]
        return coor

    def _get_position(self, base_img):
        detect_res = self._get_detection_obj365(base_img)
        df_res = detect_res.pandas().xyxy[0]
        bboxs_list = []
        for i in range(len(df_res)):
            bboxs_list.append(df_res.iloc[i, :4].tolist())
        h, w, _ = cv2.imread(base_img).shape
        while True:
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            for item in bboxs_list:
                if x < int(item[0]) or x > int(item[2]) or y < int(item[1]) or y > int(item[3]):
                    return (x, y)

    def _get_segmentation_mask(self, obj_img_path, box):
        img = cv2.imread(obj_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_box = np.array(box)
        predictor.set_image(img)
        mask, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        res = np.full_like(mask, fill_value=1, dtype=np.int16)
        res = res * mask
        res = np.transpose(res, (1, 2, 0))
        return res

    def copy_paste(self, base_img, obj_img):
        left_upper = self._get_position(base_img)
        obj_bbox = self._get_most_relevent_object_bbox(base_img, obj_img)
        obj_mask = self._get_segmentation_mask(obj_img, obj_bbox)
        h_obj, w_obj, _ = obj_mask.shape

        return obj_mask