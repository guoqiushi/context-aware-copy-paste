from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from segment_anything import SamPredictor,sam_model_registry
from PIL import Image
import random
import torch
import glob
import cv2
import os
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
        # self.obj365_labels = self._get_obj365_labels()

    def copy_paste(self, base_image_path, margin):
        category = self._get_most_relevent_category(base_image_path)
        print(category)
        target_img_list = glob.glob('D:/dataset/obj365/{}/*.png'.format(category)) + glob.glob(
            'D:/dataset/obj365/{}/*.jpg'.format(category))
        target_img_path = random.choice(target_img_list)
        target_mask, bbox = self._get_segmentation_given_category(target_img_path)
        x1, y1, x2, y2 = bbox
        target_img = cv2.imread(target_img_path)[y1:y2, x1:x2]
        target_mask = target_mask[y1:y2, x1:x2]
        base_img = cv2.imread(base_image_path)
        h_b, w_b, _ = base_img.shape
        h_t, w_t, _ = target_img.shape
        upper_left = (random.randint(margin, w_b - margin), random.randint(margin, h_b - margin))
        print(upper_left)
        max_h = h_b - upper_left[1] - 50
        max_w = w_b - upper_left[0] - 50

        resized_target_img = cv2.resize(target_img, (max_w, max_h))
        resized_target_mask = cv2.resize(target_mask, (max_w, max_h))
        print(resized_target_img.shape)
        base_mask = np.zeros_like(base_img)
        for i in range(max_h):
            for j in range(max_w):
                if resized_target_mask[i, j] == 1:
                    base_img[upper_left[1] + i, upper_left[0] + j] = resized_target_img[i, j]
                    base_mask[upper_left[1] + i, upper_left[0] + j] = 1
        return base_img, base_mask

    def _get_segmentation_given_category(self, img_path):
        x1, y1, x2, y2 = self._get_bbox_given(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_box = np.array([x1, y1, x2, y2])
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
        return res, [x1, y1, x2, y2]

    def _get_bbox_given(self, img_path):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        res = self.yolo(img_path)
        res = res.pandas().xyxy[0]
        if len(res) == 0:
            return [0, 0, h, w]
        # target_df = res[res['name']==category]
        target_df = res[res['confidence'] > 0.9]
        if len(target_df) == 0:
            target_df = res
        # index = random.randint(0,len(target_df))
        # return target_df
        return [int(x) for x in target_df.iloc[0, :].tolist()[:4]]

    def _get_most_relevent_category(self, img_path):
        caption = self._get_img_caption(img_path)
        categories = os.listdir('D:/dataset/obj365/')
        category_score = {}
        for category in categories:
            category_score[category] = self._get_similarity(category, caption)
        return sorted(category_score.items(), key=lambda x: x[1], reverse=True)[0][0]

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

    # def _get_obj365_labels(self):
    #     with open('Objects365.yaml','r',encoding='utf-8') as file:
    #         labels = yaml.load(file,Loader=yaml.FullLoader)
    #     obj365_labels = list(labels['names'].values())
    #     return obj365_labels

    # def _get_most_relevent_object_bbox(self,base_img,obj_img):
    #     similarity_res = []
    #     obj_res = self._get_detection_obj365(obj_img).pandas().xyxy[0]
    #     base_img_caption = self._get_img_caption(base_img)
    #     for i in range(len(obj_res)):
    #         similarity_res.append([i,self._get_similarity(base_img_caption,obj_res.iloc[i,-1])])
    #     index = sorted(similarity_res,key=lambda x: x[1],reverse=True)[0][0]
    #     # print(index)
    #     coor = [int(x) for x in  obj_res.iloc[index].tolist()[:4]]
    #     return coor

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





