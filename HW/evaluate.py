import cv2
import numpy as np
import pickle
import os
import os.path as osp

def rbf(val,sigma=0.1):
    return np.exp(-val/sigma)

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area1 = np.maximum(inter_rect_x2 - inter_rect_x1 + 1, 0)
    inter_area2 = np.maximum(inter_rect_y2 - inter_rect_y1 + 1, 0)
    inter_area = inter_area1*inter_area2
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def evaluate(predictions):

    tFile = "HW/annotations.pickle"

    if not osp.exists(tFile):
        print("Error: The file " + tFile + "does not exist")
        return -1

    file = open(tFile,"rb")
    targets = pickle.load(file)

    imgCnt = len(targets.keys())

    # Variables for task 1
    nObj = 0
    nCorr = 0
    nPred = 0

    # Variables for task 1 HC
    nVAndC = 0
    nCorrVandC = 0

    # Variables for task 2
    nTS = 0
    nCorrTS = 0

    # Variables for task 2 HC
    nTSExtra = 0
    nCorrTSExtra = 0

    # Variables for task 3
    coordError = 0

    # Variables for task 3 HC
    poseError = 0

    for k in targets.keys():

        target = targets[k]

        if k not in predictions.keys():
            print("Error: The image " + k + " does not exist in the predictions")
            return -1

        prediction = predictions[k]

        if 'poses' not in prediction.keys() or 'objects' not in prediction.keys():
            print("Error: The prediction for an image has to contain keys 'poses' and 'objects'")
            return -1

        tPose = target['poses']
        pPose = prediction['poses']

        tObjects = np.array(target['objects'])
        pObjects = np.array(prediction['objects'])

        '''bbNoise = np.random.randn(pObjects.shape[0],4)
        pObjects[:,:4] += bbNoise*0
        posNoise = np.random.randn(pObjects.shape[0],3)
        pObjects[:,6:] += posNoise*0
        poseNoise = np.random.randn(12)
        pPose += poseNoise*0'''

        # Task 3 HC: Reprojection error
        poseError += rbf(((tPose - pPose) ** 2).sum(), sigma=0.5)

        # Task 1: Recall, precision using IoU and main classes
        nObj += tObjects.shape[0]
        nPred += pObjects.shape[0]

        for obj in tObjects:

            # Compute IoUs
            IoUs = bbox_iou(obj,pObjects)

            # Get best box
            idx = np.argmax(IoUs)
            bestIoU = IoUs[idx]

            # If IoU and class are good
            if bestIoU > 0.25 and obj[4] == pObjects[idx,4]:

                # Correct predictions
                nCorr += 1

                # Euclidean coord distance
                coordError += rbf(((obj[6:] - pObjects[idx,6:]) ** 2).sum(), sigma=0.05)

                # Count correctly detected traffic signs
                if pObjects[idx,4] == 0:
                    if pObjects[idx, 5] < 52:
                        nTS += 1
                    else:
                        nTSExtra += 1

                # If secondary class is good
                if pObjects[idx, 5] == obj[5]:

                    # Increment counters based on the class
                    if obj[4] > 0:
                        nCorrVandC += 1
                    else:
                        if obj[5] < 52:
                            nCorrTS += 1
                        else:
                            nCorrTSExtra += 1

        # Task 1 HC: Classification accuracy of vehicle and cactus
        VNCObjects = np.array([obj for obj in tObjects if obj[4] > 0])
        nVAndC += VNCObjects.shape[0]

    recall = nCorr/nObj if nObj else 1
    precision = nCorr/nPred if nPred else 1

    task1Score = (recall + precision)/2
    task1HCScore = nCorrVandC/nVAndC

    task2Score = nCorrTS/nTS if nTS else 0
    task2HCScore = nCorrTSExtra/nTSExtra if nTSExtra else 0

    task3Score = coordError/nCorr
    task3HCScore = poseError/imgCnt

    print("Task 1:", task1Score)
    print("Task 1 HC:", task1HCScore)

    print("Task 2:", task2Score)
    print("Task 2 HC:", task2HCScore)

    print("Task 3:", task3Score)
    print("Task 3 HC:", task3HCScore)

    print("Total: ", task1Score+task1HCScore+task2HCScore+task2Score+task3HCScore+task3Score)

if __name__ == '__main__':

    pFile = "HW/annotations.pickle"

    if not osp.exists(pFile):
        print("Error: The file " + pFile + " does not exist")
        exit(-1)

    file = open(pFile, "rb")
    predictions = pickle.load(file)

    evaluate(predictions)