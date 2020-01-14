import cv2
import time

model = cv2.dnn.readNetFromCaffe(
    "model/pose_deploy_linevec.prototxt",
    "model/pose_iter_440000.caffemodel"
)


def draw_pose_line(frame, p1, p2):
    if p1 is not None and p2 is not None:
        cv2.line(frame, p1, p2, (0, 255, 255), 2)


y_list = []

while True:
    time.sleep(0.1)


    if frame is None:
        continue
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (168, 168), (0, 0, 0), swapRB=False, crop=False)
    model.setInput(inpBlob)
    output = model.forward()

    for n in range(1):
        points = []
        H = output.shape[2]
        W = output.shape[3]
        for i in range(18 - 4):
            probMap = output[n, i, :, :]
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(probMap)
            x = float(frame.shape[1]) * (maxLoc[0] + 0.5) / W
            y = float(frame.shape[0]) * (maxLoc[1] + 0.5) / H
            if maxVal >= 0.6:
                points.append((int(x), int(y)))
                y_list.append(int(y))
            else:
                points.append(None)

        for i, p in enumerate(points):
            if p is not None:
                cv2.circle(frame, p, 6, (0, 0, 255), -1)
            print(points[0])
        draw_pose_line(frame, points[0], points[1])
        draw_pose_line(frame, points[1], points[2])
        draw_pose_line(frame, points[2], points[3])
        draw_pose_line(frame, points[3], points[4])
        draw_pose_line(frame, points[1], points[5])
        draw_pose_line(frame, points[5], points[6])
        draw_pose_line(frame, points[6], points[7])
        draw_pose_line(frame, points[1], points[8])
        draw_pose_line(frame, points[8], points[9])
        draw_pose_line(frame, points[9], points[10])
        draw_pose_line(frame, points[1], points[11])
        draw_pose_line(frame, points[11], points[12])
        draw_pose_line(frame, points[12], points[13])
        # draw_pose_line(frame, points[0], points[14])
        # draw_pose_line(frame, points[14], points[16])
        # draw_pose_line(frame, points[0], points[15])
        # draw_pose_line(frame, points[15], points[17])

    print(y_list)
    if len(y_list) != 0:
        a = max(y_list)
        b = min(y_list)
        mid = a - b
        print('max', a, 'min', b, 'range', mid)
        file = open("testfile.txt", "w")
        file.write(str(mid))
        file.close()
        y_list.clear()
        print('clear')

    frame = cv2.resize(frame, (600, 480))
    cv2.imshow("frame", frame)

    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break

cv2.destroyAllWindows()
