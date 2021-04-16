"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import numpy as np
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from config import AOD_config as AOD_config

from utils.yolo_with_plugins import TrtYOLO

import paho.mqtt.client as mqtt


WINDOW_NAME = 'TrtYOLODemo'


def on_connect(client, userdata, flag, rc):
    print("Connected with result code " + str(rc))

def on_disconnect(client, userdata, flag, rc):
    if rc != 0:
        print("Unexpected disconnection.")

def on_publish(client, userdata, mid):
    print("publish: {0}".format(mid))

def init_mqtt(host, port=1883):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    client.connect(host, port, 60)
    client.loop_start()
    return client


def publish_bboxes(boxes, confs, clss, cls_dict, \
        client, topic):
    """publish detected bounding boxes to MQTT broker."""
    for bb, cf, cl in zip(boxes, confs, clss):
        print("bb_MQTT:",bb)
        #print("cf:",cf)
        print("cl:",cl)
        cl = int(cl)
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        cls_name = cls_dict.get(cl, 'CLS{}'.format(cl))
        txt = '{0}, {1:.2f}, {2}, {3}, {4}, {5}'.format(
            cls_name,
            cf,
            x_min,
            y_min,
            x_max,
            y_max
        )
        print(txt)
        client.publish(topic, txt)


def publish_bboxes_New(row_centroid_classes_metrics, state, client, topic):
    """publish detected bounding boxes to MQTT broker."""
    print("row_centroid_classes_metrics:" ,row_centroid_classes_metrics)
    x_min, y_min = row_centroid_classes_metrics[0], row_centroid_classes_metrics[1]
    x_max, y_max = row_centroid_classes_metrics[2], row_centroid_classes_metrics[3]
    cls_name = row_centroid_classes_metrics[6]
    object_ID = row_centroid_classes_metrics[-1]
    '''
    txt = '{0}, {1}, {2}, {3}, {4}'.format(
            cls_name,
            x_min,
            y_min,
            x_max,
            y_max
    )
    '''
    txt = '{}, {}, {}'.format(cls_name, state, object_ID)
    print(txt)
    client.publish(topic, txt)


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument('--host', \
        type=str, default='localhost', metavar='MQTT_HOST', \
        help='MQTT remote broker IP address')
    parser.add_argument('--topic', \
        type=str, default='yolo', metavar='MQTT_TOPIC', \
        help='MQTT topic to be published on')
    parser.add_argument('--port', \
        type=int, default=1883, metavar='MQTT_PORT', \
        help='MQTT port number')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis, \
        cls_dict, client, topic):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    frame_id = 0
    frame_id_holder = []
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        print("boxes:",boxes)
        print("confs:",confs)
        print("clss:",clss)
        #print(len(confs))
        #for cl in clss:
            #print("clss:",cl)
        #print("clss:",clss)
        #publish_bboxes(boxes, confs, clss, cls_dict, client, topic)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)

        ## wasuremono logic
        # no detected person and object or one or more persons is only detected
        # do not perform abandoned object detection
        if len(clss)==0 or ((len(clss) > 0) and (sum(clss)==0)):
            #print("length_confs:",len(confs))
            print("no detected person and object or person is only detected")
            pass
 
        # only objects are detected
        # perform abandoned object detection
        elif 0 not in clss:
            print("only objects are detected")
            if (24 in clss) or (25 in clss) or (26 in clss) or (28 in clss) or (67 in clss):
                ## Here needs to be added the function of the abandonment ##
                add_centroid_classes_metrics2 = compute_pixel_distance(boxes, clss, cls_dict)
                #print("add_centroid_classes_metrics:", add_centroid_classes_metrics2)
                for row in add_centroid_classes_metrics2:
                    if (row[6]=='backpack') or (row[6]=='umbrella') or (row[6]=='handbag') or \
                        (row[6]=='suitcase') or (row[6]=='cell phone'):
                        x_min2 = int(float(row[0]))
                        y_min2 = int(float(row[1]))
                        x_max2 = int(float(row[2]))
                        y_max2 = int(float(row[3]))
                        
                        print("add_centroid_classes_metrics2:", row)
                        color = (0,0,255)
                        cv2.rectangle(img, (x_min2,y_min2), (x_max2,y_max2), color, 2)
                        cv2.putText(img, 'abandoned', (x_min2 + 1, y_min2 - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        publish_bboxes_New(row, 'abandoned', client, topic)


        # one or more persons and objects are detected
        # perform abandoned object detection
        # except the center of object is inside a detected person's bounding box
        else:
            print("one or more persons and objects are detected")
            ## Here needs to be added the function of the abandonment ##
            #compute_pixel_per_metric_ratio(boxes, clss, cls_dict)
            #extract_pixel_per_metric_ratio = compute_pixel_per_metric_ratio(boxes, clss, cls_dict)
            #print("extract_pixel_per_metric_ratio:",extract_pixel_per_metric_ratio[-2][-2:])
            add_centroid_classes_metrics = compute_pixel_distance(boxes, clss, cls_dict)
            print()
            for row in add_centroid_classes_metrics:
                if 'person' == row[6]:
                    coord_person_min_x = int(float(row[0]))
                    coord_person_min_y = int(float(row[1]))
                    coord_person_max_x = int(float(row[2]))
                    coord_person_max_y = int(float(row[3]))
                    for objectX in add_centroid_classes_metrics:
                        if 'person' != objectX[6]:
                            centroid_object_x = int(float(objectX[4]))
                            centroid_object_y = int(float(objectX[5]))

                            if (coord_person_min_x < centroid_object_x < coord_person_max_x) and \
                                    (coord_person_min_y < centroid_object_y < coord_person_max_y):
                                        pass
                            else:
                                dis_min = 100000
                                for temp_min in objectX[7:len(add_centroid_classes_metrics)+6]:
                                    if float(temp_min) < float(dis_min):
                                        dis_min = float(temp_min)
                                
                                print()
                                print("dis_min:", dis_min)

                                if dis_min > 300:
                                    print("")
                                    print("objectX:",objectX)
                                    print("wasuremono no kanouseiari")
                                    if (frame_id > (fps * AOD_config.INITIAL_INSPECTION_DURATION)):

                                        frame_id_holder.append(frame_id)

                                        if (objectX[6]=='backpack') or (objectX[6]=='umbrella') or (objectX[6]=='handbag') or \
                                            (objectX[6]=='suitcase') or (objectX[6]=='cell phone'):
                                            # (B, G, R)
                                            color = (0,255,255)

                                            x_min2 = int(float(objectX[0]))
                                            y_min2 = int(float(objectX[1]))
                                            x_max2 = int(float(objectX[2]))
                                            y_max2 = int(float(objectX[3]))
                                            
                                            #cv2.rectangle(img, (100,100), (200,200), color, 2)

                                            cv2.rectangle(img, (x_min2,y_min2), (x_max2,y_max2), color, 2)
                                            cv2.putText(img, 'warning', (x_min2 + 1, y_min2 - 2),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                            print("frame_id_holder:", frame_id_holder)
                                            if frame_id >= (fps * ((min(frame_id_holder) / fps) + AOD_config.ABANDONMENT_DURATION)):

                                                color = (0,0,255)
                                                cv2.rectangle(img, (x_min2,y_min2), (x_max2,y_max2), color, 2)
                                                cv2.putText(img, 'abandoned', (x_min2 + 1, y_min2 - 2),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                                publish_bboxes_New(objectX, 'abandoned', client, topic)

        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        print("fps:", fps)
        tic = toc
        frame_id += 1
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)



def compute_pixel_per_metric_ratio(boxes, clss, cls_dict):
    add_pixel_per_metric_ratio = []
    for bb, cl in zip(boxes,clss):
        x_min, x_max = bb[0], bb[2]
        object_pixel_width = x_max - x_min
        cl = int(cl)
        cls_name = cls_dict.get(cl, 'CLS{}'.format(cl))
        # select appropriate object width base on the object class
        if 28 in clss:
            object_width = AOD_config.SUITCASE_WIDTH
            print("1:",object_width)
        elif 24 in clss:
            object_width = AOD_config.BACKPACK_WIDTH
            print("2:",object_width)
        else:
            object_width = AOD_config.BACKPACK_WIDTH
            print("3:",object_width)
        # get the pixel-per-metric ratio
        pixel_per_metric = object_pixel_width / object_width
        bb = np.append(bb, pixel_per_metric)
        bb = np.append(bb, cls_name)
        add_pixel_per_metric_ratio.append(bb)
        print("get the pixel-per-metric ratio:",pixel_per_metric)
        print("bb_list:",bb)
        ## Maybe here, I need to make the list ,such as pixel_per_metric.append() ##
    print("renewal_boxes2:",add_pixel_per_metric_ratio)

    return add_pixel_per_metric_ratio


def compute_pixel_distance(boxes, clss, cls_dict):
    add_centroid_classes = []
    for bb, cl in zip(boxes,clss):
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        centroid_X = (x_min + x_max) / 2
        centroid_Y = (y_min + y_max) / 2

        #add_centroid_classes_tolist = bb.tolist()
        bb = np.append(bb, centroid_X)
        bb = np.append(bb, centroid_Y)
        cl = int(cl)
        cls_name = cls_dict.get(cl, 'CLS{}'.format(cl))
        #add_centroid_classes_tolist.append(centroid_X)
        #add_centroid_classes_tolist.append(centroid_Y)
        #add_centroid_classes_tolist.append(cls_name)
        #bb = np.append(bb, cl)
        bb = np.append(bb, cls_name)
        add_centroid_classes.append(bb)
        print("bb_list:",bb)
    #print("renewal_boxes:",boxes)
    print("renewal_boxes2:",add_centroid_classes)
    
    #print("renewal_boxes3:",add_centroid_classes_tolist)

    #print("add_centroid_classes[0][0]:",add_centroid_classes[0][0])
    #print("add_centroid_classes[-2][-1]:",add_centroid_classes[-2][-1])
    #print("add_centroid_classes[-2][-3]:",add_centroid_classes[-2][-3],add_centroid_classes[-2][-2])
    #print("add_centroid_classes[0:][-3:]:",add_centroid_classes[0:][-3:])
    #print("Show_add_centroid_classes_tolist:",add_centroid_classes_tolist)

    ## compute amang persons and objects,persons and persons,objects and objects ##
    add_centroid_classes_metrics = []
    row_idx = 0
    for row in add_centroid_classes:
        cen_x = float(row[4])
        cen_y = float(row[5])
        print("X,Y:", cen_x,cen_y)
        
        row_idx2 = 0
        for row2 in add_centroid_classes:
            if (row[6] != row2[6]) and (row[6] == 'person' or row2[6] == 'person'):
                obj_cen_x = float(row2[4])
                obj_cen_y = float(row2[5])
                dist = np.sqrt((cen_x - obj_cen_x)**2 + (cen_y - obj_cen_y)**2)
                row = np.append(row, dist)
                print("x,y:", obj_cen_x,obj_cen_y)
                print("distance:",dist )
            else:
                row = np.append(row, 100000)

            #row_idx2 += 1

        row = np.append(row, row_idx)
        row_idx += 1
        add_centroid_classes_metrics.append(row)
        print("object_table_element:", row)
    print("obj_table:",add_centroid_classes_metrics)
    return add_centroid_classes_metrics
    




def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    client = init_mqtt(args.host, args.port)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    print("cls_dict:",cls_dict)
    #print(cls_dict[3])
    yolo_dim = args.model.split('-')[-1]
    print("yolo_dim:",yolo_dim)
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
        print('w:{0}, h:{1}'.format(w,h))
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis, \
        cls_dict=cls_dict, client=client, topic=args.topic)

    cam.release()
    cv2.destroyAllWindows()

    client.disclose()


if __name__ == '__main__':
    main()
