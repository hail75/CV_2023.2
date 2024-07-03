import sys
import pickle as pkl
import threading
import time

import cv2
import numpy as np
import torch
import torch.quantization
import scipy.special

sys.path.append("..")
import models
from utils.misc import to_torch
from utils.imutils import im_to_numpy, im_to_torch, resize_generic
from utils.transforms import color_normalize


gloss_show = ''

def main():
    dim = (224, 224)
    frames = 16
    channels = 3
    model_path = "../data/experiments/wlasl_i3d_pbsl1k/model.pth.tar"
    threshold = 50

    frame_buffer = np.empty((0, *dim, channels))

    model = models.InceptionI3d(
        num_classes=2000,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=frames,
    )

    model.eval()

    for module_name, module in model.named_children():
        if module_name == "Logits" or "_" in module_name:
            for block_name, block in module.named_children():
                if block_name == 'Unit3D':
                    torch.quantization.fuse_modules(block, ['Conv3d', 'BatchNorm3d', 'ReLU'], inplace=True)
                elif block_name == 'InceptionModule':
                    for sub_block_name, sub_block in block.named_children():
                        if sub_block_name == 'Unit3D':
                            torch.quantization.fuse_modules(sub_block, ['Conv3d', 'BatchNorm3d', 'ReLU'], inplace=True)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(model_path)
    else:
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["state_dict"])
    

    cap = cv2.VideoCapture(0)
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_FPS, 25)

    frame_buffer = np.empty((0, channels, cap_height, cap_width))

    x = threading.Thread()
    prev_frame_time = 0
  
    new_frame_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:      
            frame_res = frame[:, :, [2, 1, 0]]
            frame_res = im_to_torch(frame_res).numpy()

            frame_resh = np.reshape(frame_res, (1, *frame_res.shape))
            frame_buffer = np.append(frame_buffer, frame_resh, axis=0)


            if frame_buffer.shape[0] == frames:
                if not x.is_alive():
                    x = threading.Thread(target=make_prediction, args=(
                        frame_buffer, model, threshold))
                    x.start()
                else:
                    pass                
                frame_buffer = frame_buffer[1:frames]

                font = cv2.FONT_HERSHEY_DUPLEX
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time) 
                prev_frame_time = new_frame_time 
                fps_show = f'FPS: {int(fps)}'
                cv2.putText(frame, fps_show, (7, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, gloss_show, (20, 450), font, 1, (0, 255, 0),
                            2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def make_prediction(frame_buffer, model, threshold):
    global gloss_show
    with torch.inference_mode():
        frame_buffer = torch.from_numpy(frame_buffer).permute(1, 0, 2, 3)

        resize_res = 224
        mean = 0.5 * torch.ones(3)
        std = 1.0 * torch.ones(3)

        iC, iF, iH, iW = frame_buffer.shape
        frame_buffer_resized = np.zeros((iF, resize_res, resize_res, iC))
        for t in range(iF):
            tmp = frame_buffer[:, t, :, :]
            tmp = resize_generic(
                im_to_numpy(tmp), resize_res, resize_res, interp="bilinear", is_flow=False
            )
            frame_buffer_resized[t] = tmp

        inp = np.transpose(frame_buffer_resized, (3, 0, 1, 2))
        inp = to_torch(inp).float()
        inp = color_normalize(inp, mean, std)

        inp = inp.unsqueeze(0)
        
        out = model(inp)        
        # print( profile(model, inputs=(inp,)))   
        # flops, params = get_model_complexity_info(model, (3, 16, 224, 224), as_strings=True, print_per_layer_stat=False)

        raw_scores = np.empty((0, 2000), dtype=float)
        out_reshaped = out.cpu().detach().numpy().mean(axis=(2, 3))

        raw_scores = np.append(raw_scores, out_reshaped, axis=0)
        prob_scores = scipy.special.softmax(raw_scores, axis=1)
        word_data = pkl.load(open("../data/wlasl/info/info.pkl", "rb"))
        top_prediction_index = np.argmax(prob_scores, axis=1)[0]
        top_prediction_word = word_data["words"][top_prediction_index]
        top_prediction_confidence = prob_scores[0, top_prediction_index] * 100

        if top_prediction_confidence > threshold:
            gloss_show = f'{top_prediction_word} ({top_prediction_confidence:.2f}%)'
        else:
            gloss_show = ''

if __name__ == "__main__":  
    main()

