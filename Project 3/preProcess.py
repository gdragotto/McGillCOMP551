import gzip
import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import cv2
import imutils




def visualize(point, label):
    pixels = np.array(point, dtype='uint8')
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels)
    plt.show()


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (img.shape[0], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def treshold(img):
    im = np.array(img, dtype=np.uint8)
    im = cv2.threshold(im, 250, 255, cv2.THRESH_TOZERO)[1]
    im.astype('uint8')
    return im


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def processDigit(img, label):
    #visualize(img, label)
    im = treshold(img)
    #visualize(im, label)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dA = []
    sP = []
    H = []
    W = []
    i = 0
    for item in contours:
        x, y, w, h = cv2.boundingRect(item)
        orig = im.copy()
        box = cv2.minAreaRect(item)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        d1 = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        d2 = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        #print("box " + str(i) + ": " + str(d1) + "-" + str(d2))
        #print("\tcoord x="+str(x)+" y="+str(y))
        new = np.zeros(im.shape)
        new = cv2.rectangle(new, (x, y), (x + w, y + h), (255, 255, 255), -1)
        #visualize(new, label)
        dA.append(2*max(d1,d2)+d1+d2)
        H.append(h)
        W.append(w)
        i += 1

    index = np.argmax(dA)
    #print("got " + str(index)+" with "+str(dA[index]))
    x, y, w, h = cv2.boundingRect(contours[index])
    mDim = max(H[index],W[index])
    cx = x + mDim / 2
    cy = y + mDim / 2
    new = np.zeros(im.shape, dtype=np.uint8)
    new = cv2.rectangle(new, (x, y), (x + mDim, y + mDim), (255, 255, 255), -1)
    #visualize(new, label)
    preprocessed_img = np.zeros(im.shape, dtype=np.uint8)
    preprocessed_img = im*new
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(preprocessed_img, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255.
    cx = 32 - centroids[max_label][0]
    cy = 32 - centroids[max_label][1]
    M = np.float32([[1, 0, cx], [0, 1, cy]])
    dst = cv2.warpAffine(img2, M, (64, 64))
    #dst = deskew(dst[18:46, 18:46])
    dst = dst[18:46, 18:46]
    #visualize(dst, label)
    return dst

with open( 'train_images.pkl', 'rb') as f:
    train = pickle.load(f)
with open('test_images.pkl', 'rb') as f:
    test = pickle.load(f)
labels = []

with open( "train_labels.csv") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    first = True
    for row in reader:
        labels.append(row)
data = []
for i in range(len(labels)):
    point = np.zeros(28 * 28 + 2)
    # Image Id
    point[0] = labels[i][0]
    # Label
    point[1] = labels[i][1]
    processd = processDigit(train[i], labels[i][1]).flatten()
    for index, item in enumerate(processd):
        point[index + 2] = item
    data.append(point)
    if (i % 1000 == 0):
        print("Pre-processing trainingSet item " + str(i) + "...")
        # visualize(processDigit(train[i]), labels[i][1])

np.save('trainingSet.npy', data)

data = []
for i in range(len(test)):
    point = np.zeros(28 * 28 + 1)
    point[0] = i
    processd = processDigit(test[i],0).flatten()
    for index, item in enumerate(processd):
        point[index + 1] = item
    data.append(point)
    if (i % 1000 == 0):
        print("Pre-processing testSet item " + str(i) + "...")
    # visualize(processd, labels[i][1])
np.save('testSet.npy', data)
