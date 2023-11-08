import os.path

from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


def coordinate2yolo(bbox, img_h, img_w):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]
    x = round((x_min + x_max) / (2.0 * img_w), 6)
    y = round((y_min + y_max) / (2.0 * img_h), 6)
    w1 = round((x_max - x_min) / (1.0 * img_w), 6)
    h1 = round((y_max - y_min) / (1.0 * img_h), 6)
    return [x, y, w1, h1]


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        if bboxes.numel() == 0:
            # if no bounding box
            print(f'{image}: no object!')
            continue
        bboxes -= dwdh
        bboxes /= ratio
        file_name = os.path.splitext(save_image)[0] + ".txt"
        with open(file_name, 'a+') as label_file:
            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                color = COLORS[cls]
                # cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                cv2.putText(draw,
                            f'{cls}:{score:.3f}', (bbox[2], bbox[3] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255],
                            thickness=2)
                label_ = coordinate2yolo(bbox, draw.shape[0], draw.shape[1])
                label_file.write(f"{cls_id} {label_[0]} {label_[1]} {label_[2]} {label_[3]}\n")
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='/home/rp/DYH/yolov8/ultralytics/test_val',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
