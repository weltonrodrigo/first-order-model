import warnings
from argparse import ArgumentParser

import face_alignment
import imageio
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize

warnings.filterwarnings("ignore")


def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor = max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor


def compute_bbox(tube_bbox, frame_shape, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    # Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])

    return (left, top, right, bot)


def process_video(args):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                      device='cpu' if args.cpu else 'cuda')
    image = imageio.imread(args.inp)

    bboxes = extract_bbox(image, fa)

    image_shape = image.shape[:2]

    (left, top, right, bottom) = compute_bbox(bboxes[0], image_shape, args.increase)

    resized = resize(image[top:bottom, left:right], (256, 256))

    return resized


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--inp", required=True, help='Input image or video')
    parser.add_argument("--output", required=True, help='Output image')
    parser.add_argument("--cpu", dest="cpu", default=False, action="store_true", help="cpu mode.")

    args = parser.parse_args()

    cropped = process_video(args)

    try:
        imageio.imsave(args.output, cropped)
    except Exception as e:
        raise (Exception)
