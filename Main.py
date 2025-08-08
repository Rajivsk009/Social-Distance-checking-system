

from tkinter import *
import os
from tkinter import filedialog
import cv2
import time

from tkinter import messagebox





def endprogram():
	print ("\nProgram terminated!")
	sys.exit()










def fulltraining():
    import Distance as mm
















def imgtraining():


    import_file_path = filedialog.askopenfilename()
    import os
    s = import_file_path
    os.path.split(s)
    os.path.split(s)[1]
    splname = os.path.split(s)[1]

    #image = cv2.imread(import_file_path)

    import numpy as np
    import time
    import cv2
    import math

    labelsPath = "data/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    print(LABELS)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    weightsPath = "data/yolov3.weights"
    configPath = "data/yolov3.cfg"

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    cap = cv2.VideoCapture(import_file_path)
    flagg = 1
    while (cap.isOpened()):

        ret, image = cap.read()
        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        # print("Frame Prediction Time : {:.6f} seconds".format(end - start))
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.1 and classID == 0:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        ind = []
        for i in range(0, len(classIDs)):
            if (classIDs[i] == 0):
                ind.append(i)
        a = []
        b = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                a.append(x)
                b.append(y)

        distance = []
        nsd = []
        for i in range(0, len(a) - 1):
            for k in range(1, len(a)):
                if (k == i):
                    break
                else:
                    x_dist = (a[k] - a[i])
                    y_dist = (b[k] - b[i])
                    d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                    distance.append(d)
                    if d <= 100:
                        nsd.append(i)
                        nsd.append(k)
                    nsd = list(dict.fromkeys(nsd))
                    # print(nsd)
        color = (0, 0, 255)
        for i in nsd:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "Alert"
            flagg += 1
            # print(flag)

            if (flagg == 50):
                flagg = 0
                import winsound

                filename = 'alert.wav'
                winsound.PlaySound(filename, winsound.SND_FILENAME)

            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        color = (0, 255, 0)
        if len(idxs) > 0:
            for i in idxs.flatten():
                if (i in nsd):
                    break
                else:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = 'OK'
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Social Distancing Detector", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()






def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    main_screen.title("Social Distancing Detector")

    Label(text="Social Distancing Detector", width="300", height="5", font=("Calibri", 16)).pack()


    Button(text="Upload Video", font=(
        'Verdana', 15), height="2", width="30", command=imgtraining, highlightcolor="black").pack(side=TOP)

    Label(text="").pack()


    Label(text="").pack()

    main_screen.mainloop()


main_account_screen()

