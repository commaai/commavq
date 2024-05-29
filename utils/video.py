import cv2
import numpy as np

CROP_SIZE = (512, 256)
OUTPUT_SIZE = (CROP_SIZE[0]//2, CROP_SIZE[1]//2)
SCALE = 567 / 455
CY = 47.6

def write_video(frames_rgb, out, fps=20):
  size = frames_rgb[0].shape[:2][::-1]
  video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'avc1'), fps, size)
  for frame in frames_rgb:
    video.write(frame[...,::-1])
  video.release()
  return out

def read_video(path):
  frames = []
  cap = cv2.VideoCapture(path)
  ret = True
  while ret:
    ret, img = cap.read()
    if ret:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      frames.append(img)
  video = np.stack(frames, axis=0)
  return video

def transpose_and_clip(tensors):
  tensors = np.array(tensors)
  tensors = np.transpose(tensors, (0,2,3,1))
  tensors = np.clip(tensors, 0, 255).astype(np.uint8)
  return tensors

def transform_img(frame, output_size=OUTPUT_SIZE, crop_size=CROP_SIZE, scale=SCALE, cy=CY):
  size = frame.shape[:2][::-1]
  scaled_crop_size = (int(crop_size[0]*scale), int(crop_size[1]*scale))
  x0, y0 = size[1]//2 - int(crop_size[1]*scale)//2 - int(cy*scale)//2, size[0]//2 - int(crop_size[0]*scale)//2
  frame = frame[x0:x0+scaled_crop_size[1], y0:y0+scaled_crop_size[0]]
  return cv2.resize(frame, output_size)
