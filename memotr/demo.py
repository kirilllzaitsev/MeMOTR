import time
import cv2
import numpy as np


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.end_time = time.time()
        self.total_time = 0
        self.calls = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.end_time = time.time()
        self.total_time += self.end_time - self.start_time
        self.calls += 1

    @property
    def average_time(self):
        return self.total_time / max(1, self.calls)


def process_image(image):
    ori_image = image.copy()
    h, w = image.shape[:2]
    scale = 800 / min(h, w)
    if max(h, w) * scale > 1536:
        scale = 1536 / max(h, w)
    target_h = int(h * scale)
    target_w = int(w * scale)
    image = cv2.resize(image, (target_w, target_h))
    image = F.normalize(
        F.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    return image, ori_image


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0.0, ids2=None):
    # Thanks to https://github.com/noahcao/OC_SORT
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w / 140.0))
    cv2.putText(
        im,
        # "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)),
        "frame: %d" % (frame_id),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 255, 0),
        thickness=2,
    )

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = "{}".format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ", {}".format(int(ids2[i]))
        # color = get_color(abs(obj_id))
        color = (255, 255, 0)
        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
        )
        cv2.putText(
            im,
            id_text,
            (intbox[0], intbox[1]),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (255, 255, 0),
            thickness=text_thickness,
        )
    return im
