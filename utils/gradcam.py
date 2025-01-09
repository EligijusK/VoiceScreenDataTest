import cv2
import numpy as np
from PIL import Image
from utils.mfcc_image import spectrogram_image, gray_inferno
from utils.resize_image import resize_image
from pytorch_grad_cam import GradCAM as GradCAM

def get_gradcam(model, input):
    with GradCAM(model=model.model, target_layers=[model.model.model.maxpool], use_cuda=model.device != "cpu") as gcam:
        mask = np.repeat(np.array(gcam(input_tensor=input)[0] * 255, dtype=np.uint8)[..., np.newaxis], 4, -1)
        # image = gray_inferno((spectrogram_image(input[0].cpu().numpy())))
        _image = cv2.convertScaleAbs(gray_inferno(spectrogram_image(input[0].cpu().numpy())) / 255)
        image = np.full([*_image.shape[0:2], 4], 255, dtype=np.uint8)
        image[..., 0:3] = _image

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # masked = Image.blend(image, mask, 0.8)
        masked = image.copy()
        masked.paste(mask, mask=mask)

        cv2.imshow("masked image", resize_image(np.array(masked), 1000))
        cv2.waitKey(-1)
        
        return masked