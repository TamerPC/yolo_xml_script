from darkflow.net.build import TFNet
import cv2
import numpy as np
from pascal_voc_writer import Writer
from datetime import datetime as dt

def save_xml(source, save_path):
    options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.7, "gpu": 0.7}
    predictor = TFNet(options)

    capture = cv2.VideoCapture(source)

    if (capture.isOpened() == True): 
        print('script is working!')
        while(capture.isOpened()):
            ret, frame = capture.read()
            if ret == True:
                yolo_result = predictor.return_predict(frame)
                if yolo_result:
                    filename = dt.now()
                    height = np.shape(frame)[0]
                    width = np.shape(frame)[1]
                    writer = Writer(path=f'{save_path}/{filename}.jpg', width=width, height=height, database="some_db")
                    for res in yolo_result:
                        # ::addObject(name, xmin, ymin, xmax, ymax)
                        writer.addObject(res['label'], res['topleft']['x'], res['topleft']['y'], res['bottomright']['x'], res['bottomright']['y'])
                    writer.save(f'{save_path}/{filename}.xml')
                    cv2.imwrite(f'{save_path}/{filename}.jpg', frame)
        print('done')
    else:
        print('!!!error reading!!!')

if __name__ == "__main__":
    read_path = "sources/first_preson_car.mp4"
    save_path = "dataset_xmls"
    save_xml(read_path, save_path)