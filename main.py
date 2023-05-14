import cv2
import numpy as np
import pandas as pd

# Custom hardcoded camera specs
FOCAL_LENGTH = 590
SENSOR_HEIGHT = 1.344
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

model_name = 'yolov3'
model_weights = model_name + '.weights'
model_configuration = model_name + '.cfg'

net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = net.getLayerNames()
output_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def getOutputs(img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    outputs = net.forward(output_names)
    return outputs


def getCoords(outputs, hT, wT):
    confs = []
    class_ids = []
    bbox = []
    for out in outputs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    return bbox, class_ids, confs


def getIndices(bbox, confs):
    indices = cv2.dnn.NMSBoxes(bbox, confs, CONF_THRESHOLD, NMS_THRESHOLD)
    return indices


def getRealHeight(class_name):
    # Buat 80 class dengan asumsi real height nya

    return float(actual_height_list[class_name][0])


def distanceToObject(realHeight, imageHeightPixels, objectHeightPixels):
    # in mm
    return (realHeight * FOCAL_LENGTH) / objectHeightPixels
    # return (FOCAL_LENGTH * realHeight * imageHeightPixels)/ (objectHeightPixels * SENSOR_HEIGHT)


def drawImg(img, bbox, class_ids, confs):
    hT, wT, _ = img.shape
    indices = getIndices(bbox, confs)

    min_distance = np.inf
    for i in indices:
        x, y, w, h = bbox[i]
        class_name = class_names[class_ids[i]]

        # ? Calculate Object Distances
        real_height = getRealHeight(class_name) 
        distance = distanceToObject(real_height, hT, h)
        if distance < min_distance:
            min_distance = distance

        # ? Draw individual detected objects
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            img, 
            f'{class_name.upper()} {confs[i] * 100:.2f}%', 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # ? Show the closest object from the camera
    cv2.putText(
        img, 
        f'Closest Distance to Camera :  {min_distance/10:.2f} cm', 
        (0, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2
    )

    return img


def detectAndDraw(img):
    hT, wT, _ = img.shape

    outputs = getOutputs(img)

    bbox, class_ids, confs = getCoords(outputs, hT, wT)

    out_img = drawImg(img, bbox, class_ids, confs)

    return out_img

if __name__ == "__main__":
    with open('coco.names', 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    df = pd.read_csv('actual_height.csv')
    actual_height_list = df.set_index('class_name').T.to_dict('list')

    vid = cv2.VideoCapture(0)
    vid.set(3,640)
    vid.set(4,480)

    whT = 320

    while True:
        _, img = vid.read()

        out_img = detectAndDraw(img)

        # Display the resulting frame
        cv2.imshow('frame', out_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

