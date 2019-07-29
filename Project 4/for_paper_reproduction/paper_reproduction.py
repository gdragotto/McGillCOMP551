import numpy as np
from squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import os

model = SqueezeNet()


with open('../validation/ILSVRC2012_validation_ground_truth.txt') as gt:
    y = gt.read().splitlines()

map = {}
with open('../validation/map_clsloc.txt') as f:
    for line in f:
        arr = line.split(' ')
        map[arr[0]]=arr[1]


dataDir = "../validation/ILSVRC2012_img_val/"
right_1x = 0
right_5x = 0
counter = 0
output = open("ValidationLog.csv", "w")
output.write("Id,Truth,Score1,Score5,Predicted1,Predicted2,Predicted3,Predicted4,Predicted5" + "\n")
print("Validating...")
for filename in os.listdir(dataDir):
    counter+=1
    r1 = 0
    r5 = 0
    id = int(filename.replace(".JPEG","").replace("ILSVRC2012_val_",""))
    img = image.load_img(os.path.join(dataDir, filename), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds)
    for i in range(5):
        if (map[decoded[0][i][0]] == y[id-1]):
            right_5x+=1
            r5=1
            if (i == 0):
                right_1x+=1
                r1=1
    if (counter % 1000 == 0):
        v1x = right_1x/counter
        v5x = right_5x/counter
        print("\tValidating the "+str(counter)+"th example (1x: "+str(v1x)+", 5x: "+str(v5x)+")")
    output.write(str(id) + "," + str(y[id-1])+ "," + str(r1) + "," + str(r5) + "," + str(map[decoded[0][0][0]])+ "," + str(map[decoded[0][1][0]])+ "," + str(map[decoded[0][2][0]])+ "," + str(map[decoded[0][3][0]])+ "," + str(map[decoded[0][4][0]])+"\n")

v1x = right_1x/counter
v5x = right_5x/counter
print("Validation Accuracy (1x): "+str(v1x))
print("Validation Accuracy (5x): "+str(v5x))
