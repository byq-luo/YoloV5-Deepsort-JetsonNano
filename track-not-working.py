import sys
sys.path.insert(0, './yolov5')

import numpy as np
import csv
from datetime import datetime
import sqlite3
import torch.backends.cudnn as cudnn
import torch
import cv2
from pathlib import Path
import time
import shutil
import platform
import os
import argparse
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import LoadImages, LoadStreams


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def start_database(destiny):
  file_timestamp = datetime.utcnow().strftime('%d-%m-%Y[%H-%M-%S]')
  db_file = f"{file_timestamp}.db"
  if destiny == 'local':
    if not os.path.exists('database'):
      os.makedirs('database')
    db_path = os.path.join(os.getcwd(), "database", db_file)
    
  elif destiny == 'usb':
    db_path = f"/media/perceptron/DATALOG/database/{db_file}"

  db_connection = sqlite3.connect(db_path)
  db_cursor = db_connection.cursor()
  db_cursor.execute(
      '''CREATE TABLE IF NOT EXISTS classifications (
    frame INTEGER,
    class TEXT,
    detections INTEGER, 
    count INTEGER,
    date TEXT,
    bbox_top INTEGER,
    bbox_left INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER
    )'''
  )
  return db_connection, db_cursor, file_timestamp


def xyxy_to_xywh(*xyxy):
  """" Calculates the relative bounding box from absolute pixel values. """
  bbox_left = min([xyxy[0].item(), xyxy[2].item()])
  bbox_top = min([xyxy[1].item(), xyxy[3].item()])
  bbox_w = abs(xyxy[0].item() - xyxy[2].item())
  bbox_h = abs(xyxy[1].item() - xyxy[3].item())
  x_c = (bbox_left + bbox_w / 2)
  y_c = (bbox_top + bbox_h / 2)
  w = bbox_w
  h = bbox_h
  return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
  tlwh_bboxs = []
  for i, box in enumerate(bbox_xyxy):
    x1, y1, x2, y2 = [int(i) for i in box]
    top = x1
    left = y1
    w = int(x2 - x1)
    h = int(y2 - y1)
    tlwh_obj = [top, left, w, h]
    tlwh_bboxs.append(tlwh_obj)
  return tlwh_bboxs


def compute_color_for_labels(label):
  """
  Simple function that adds fixed color depending on the class
  """
  color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
  return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
  for i, box in enumerate(bbox):
    x1, y1, x2, y2 = [int(i) for i in box]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]
    # box text and bar
    id = int(identities[i]) if identities is not None else 0
    color = compute_color_for_labels(id)
    label = '{}{:d}'.format("", id)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(
        img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(img, label, (x1, y1 +
                t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
  return img


def detect(opt):
  out, source, weights, save_local_db, save_data_usb, show_vid, save_vid, save_txt, imgsz, distance_th = \
      opt.output, opt.source, opt.weights, opt.save_local_db, opt.save_data_usb, opt.show_vid, opt.save_vid, opt.save_txt, opt.img_size, opt.distance
  webcam = source == '0' or source.startswith(
      'rtsp') or source.startswith('http') or source.endswith('.txt')

  # initialize db
  # save_local_db = True
  if save_local_db:
    db_connection, db_cursor, file_timestamp = start_database('local')

  if save_data_usb:
    out = f"/media/perceptron/DATALOG/video"
    db_connection, db_cursor, file_timestamp = start_database('usb')
    csv_path = f"{Path('/media/perceptron/DATALOG/csv')}/{file_timestamp}.csv"
    with open(csv_path, 'w', encoding='UTF8') as f:
      writer = csv.writer(f)
      writer.writerow(["frame", "class", "detections", "count", "datetime"])

  distances = distance_th.split("-")
  distances = list(map(int, distances))
  amount_of_classes = len(distances)

  # initialize deepsort
  cfg = get_config()
  cfg.merge_from_file(opt.config_deepsort)
  deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=True
  )

  # Initialize
  device = select_device(opt.device)
  if os.path.exists(out):
    shutil.rmtree(out)  # delete output folder
  os.makedirs(out)  # make new output folder
  half = device.type != 'cpu'  # half precision only supported on CUDA

  # Load model
  model = attempt_load(weights, map_location=device)  # load FP32 model
  stride = int(model.stride.max())  # model stride
  imgsz = check_img_size(imgsz, s=stride)  # check img_size
  names = model.module.names if hasattr(
      model, 'module') else model.names  # get class names
  if half:
    model.half()  # to FP16

  # Set Dataloader
  vid_path, vid_writer = None, None
  # Check if environment supports image displays
  if show_vid:
    show_vid = check_imshow()

  if webcam:
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
  else:
    dataset = LoadImages(source, img_size=imgsz)

  # Get names and colors
  names = model.module.names if hasattr(model, 'module') else model.names
  for name_id in range(7):
    print(f"Clase {name_id} = {names[int(name_id)]}")

  # Run inference
  if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(
        device).type_as(next(model.parameters())))  # run once
  t0 = time.time()

  save_path = str(Path(out))
  txt_path = str(Path(out)) + '/results.txt'

  classes_count = [0] * amount_of_classes
  last_center = np.zeros((2,amount_of_classes))
  new_center = np.zeros((2,amount_of_classes))
  first_class_detection = [True] * amount_of_classes
  sort_list = np.zeros((2,amount_of_classes))
  sort_idxs = np.zeros((2,amount_of_classes), np.int32)

  for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
    # img = np.array([np.fliplr(img[i]) for i in range(3)])

    # Save datetime
    image_datetime = datetime.utcnow().strftime('%d-%m-%Y %H:%M:%S.%f')[:-2]

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
      img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(
      pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
      if webcam:  # batch_size >= 1
        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
      else:
        p, s, im0 = path, '', im0s

      s += '%gx%g ' % img.shape[2:]  # print string
      save_path = str(Path(out) / Path(p).name)
      if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(
            img.shape[2:], det[:, :4], im0.shape).round()

        xywh_bboxs = []
        confs = []

        # Adapt detections to deep sort input format
        for *xyxy, conf, cls in det:
          # to deep sort format
          x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
          xywh_obj = [x_c, y_c, bbox_w, bbox_h]
          xywh_bboxs.append(xywh_obj)
          confs.append([conf.item()])

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        # pass detections to deepsort
        outputs = deepsort.update(xywhs, confss, im0)

        tlwh_bboxs = [[]]
        # draw boxes for visualization
        if len(outputs) > 0:
          bbox_xyxy = outputs[:, :4]
          identities = outputs[:, -1]
          # to MOT format
          tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

          # Write MOT compliant results to file

          # print(outputs[-1])
          for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
            print(f"{j} --- {outputs}")
            print(f"{j} --- {outputs}")
            print(f"{j} --- {outputs}")
            print(f"{j} --- {outputs}")
            print(f"{j} --- {outputs}")
            bbox_top = tlwh_bbox[0]
            bbox_left = tlwh_bbox[1]
            bbox_w = tlwh_bbox[2]
            bbox_h = tlwh_bbox[3]
            identity = output[-1]

            # if j == 0:                        # j == 0 -> Clase 0
            new_center[0:1, identity] = tlwh_bbox[0:1]
            if first_class_detection[identity]:
              last_center[0:1, identity] = tlwh_bbox[0:1]
              first_class_detection[identity] = False
            # print(f"Last - new center = {last_center-new_center}")
            distance = np.linalg.norm(last_center[:,identity]-new_center[:,identity])
            print('Distance', distance)
            if distance > distances[identity]:
              classes_count[identity] += 1

            last_center[0:1,identity] = new_center[0:1,identity]

            if save_txt:
              with open(txt_path, 'a') as f:
                f.write(('%g ' * 10 + '\n') % (
                    frame_idx, identity+1, bbox_top, bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
            if classes_count[identity] == 0:       # Le agregué un tab
              classes_count[identity] += (j+1)

          sort_list = np.zeros(len(tlwh_bboxs))
          for pavo in range(len(tlwh_bboxs)):
            sort_list[pavo] = tlwh_bboxs[pavo][1]
          sort_idxs = np.argsort(sort_list)

          # print('sort', sort_list, sort_idxs, sort_idxs[-1])
          draw_boxes(im0, bbox_xyxy, identities)

        # Print results
        for c in det[:, -1].unique():
          class_name = names[int(c)]
          class_detections = (det[:, -1] == c).sum()  # detections per class
          current_count = classes_count[int(c)]
          s += '%g %ss, ' % (class_detections, class_name)  # add to string
          if save_local_db or save_data_usb:
            temp_bbox_top = tlwh_bboxs[0][0] if tlwh_bboxs[0] else -1
            temp_bbox_left = tlwh_bboxs[0][1] if tlwh_bboxs[0] else -1
            temp_bbox_w = tlwh_bboxs[0][2] if tlwh_bboxs[0] else -1
            temp_bbox_h = tlwh_bboxs[0][3] if tlwh_bboxs[0] else -1
            db_cursor.execute(
              "INSERT INTO classifications (frame, class, detections, count, date, bbox_top, bbox_left, bbox_w, bbox_h) VALUES (?, ?, ?, ?, ?, ?, ? ,? ,?)",
              (int(frame_idx), str(class_name), int(class_detections),
                int(current_count), str(image_datetime), temp_bbox_top, 
                temp_bbox_left, temp_bbox_w, temp_bbox_h)
            )
            db_connection.commit()
          if save_data_usb:
            with open(csv_path, 'a', encoding='UTF8') as f:
              writer = csv.writer(f)
              writer.writerow([frame_idx, class_name, class_detections, current_count, image_datetime])
          print(f"Found {class_detections} {class_name} with count {current_count} \
            at {image_datetime} in frame {frame_idx}\
            {'s' * (class_detections > 1)}")
        print('Cuentas: ', classes_count)

      else:
        deepsort.increment_ages()

      # Print time (inference + NMS)
      print('%sDone. (%.3fs)' % (s, t2 - t1))

      # Stream results
      if show_vid:
        cv2.imshow(p, im0)
        if cv2.waitKey(1) == ord('q'):  # q to quit
          raise StopIteration

      # Save results (image with detections)
      if save_vid or save_data_usb:
        if vid_path != save_path:  # new video
          vid_path = save_path
          if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()  # release previous video writer
          if vid_cap:  # video
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          else:  # stream
            fps, w, h = 30, im0.shape[1], im0.shape[0]
            save_path += f"_{file_timestamp}.mp4"

          vid_writer = cv2.VideoWriter(
              save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(im0)

  if save_txt or save_vid:
    print('Results saved to %s' % os.getcwd() + os.sep + out)
    if platform == 'darwin':  # MacOS
      os.system('open ' + save_path)

  print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights', type=str,
                      default='yolov5/weights/yolov5s.pt', help='model.pt path')
  # file/folder, 0 for webcam
  parser.add_argument('--source', type=str, default='0', help='source')
  parser.add_argument('--output', type=str, default='inference/output',
                      help='output folder')  # output folder
  parser.add_argument('--img-size', type=int, default=640,
                      help='inference size (pixels)')
  parser.add_argument('--conf-thres', type=float, default=0.4,
                      help='object confidence threshold')
  parser.add_argument('--iou-thres', type=float, default=0.5,
                      help='IOU threshold for NMS')
  parser.add_argument('--fourcc', type=str, default='mp4v',
                      help='output video codec (verify ffmpeg support)')
  parser.add_argument('--device', default='',
                      help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--save-local-db', action='store_true',
                      help='save data in sqlite3 db')
  parser.add_argument('--save-data-usb', action='store_true',
                      help='save data in mounted usb')
  parser.add_argument('--show-vid', action='store_true',
                      help='display tracking video results')
  parser.add_argument('--save-vid', action='store_true',
                      help='save video tracking results')
  parser.add_argument('--save-txt', action='store_true',
                      help='save MOT compliant results to *.txt')
  # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
  parser.add_argument('--classes', nargs='+',
                      default=[0, 1], type=int, help='filter by class')
  parser.add_argument('--agnostic-nms', action='store_true',
                      help='class-agnostic NMS')
  parser.add_argument('--augment', action='store_true',
                      help='augmented inference')
  parser.add_argument("--config_deepsort", type=str,
                      default="deep_sort_pytorch/configs/deep_sort.yaml")
  parser.add_argument('--distance', type=str, default=40,
                      help='distance between class 0 object (pixels)')
  args = parser.parse_args()
  args.img_size = check_img_size(args.img_size)

  with torch.no_grad():
    detect(args)
