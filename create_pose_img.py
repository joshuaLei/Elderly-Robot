import cv2
import numpy as np
import time

class OpenPose(object):

    def __init__(self):
        self.proto_path = "model/pose_deploy_linevec.prototxt"
        self.model_path = "model/pose_iter_440000.caffemodel"
        self.mean = (0, 0, 0)
        self.points = (
            "Nose", "Neck",                 # 0, 1
            "R-Sho", "R-Elb", "R-Wr",       # 2, 3, 4
            "L-Sho", "L-Elb", "L-Wr",       # 5, 6, 7
            "R-Hip", "R-Knee", "R-Ank",     # 8, 9, 10
            "L-Hip", "L-Knee", "L-Ank",     # 11, 12, 13
            "R-Eye", "L-Eye",               # 14, 15
            "R-Ear", "L-Ear"                # 16, 17
        )
        self.pairs = (
            (1, 0),
            (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13),
            (0, 14), (14, 16),
            (0, 15), (15, 17)
        )
        self.PAFs = (
            (47, 48),
            (31, 32), (33, 34), (35, 36),
            (39, 40), (41, 42), (43, 44),
            (19, 20), (21, 22), (23, 24),
            (25, 26), (27, 28), (29, 30),
            (49, 50), (53, 54),
            (51, 52), (55, 56)
        )
        self.colors = (
            (0, 0, 255),
            (255, 255, 255), (255, 255, 255), (255, 255, 255),
            (255, 255, 255), (255, 255, 255), (255, 255, 255),
            (255, 0, 0), (240, 255, 0), (255, 0, 240),
            (0, 255, 0), (0, 240, 255), (0, 120, 255),
            (255, 255, 255), (255, 255, 255),
            (255, 255, 255), (255, 255, 255)
        )
        self.model = cv2.dnn.readNet(self.model_path, self.proto_path)

    def detect(self, image, in_height=368, thresh=0.1):
        h, w = image.shape[:2]
        in_width = int((in_height / h) * w)

        blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (in_width, in_height), self.mean, False, False)
        self.model.setInput(blob)
        output = self.model.forward()

        auto_id = 0
        all_points = []
        point_list = []
        for i, p in enumerate(self.points):
            one_part_points = []
            prob_map = output[0, i, :, :]
            prob_map = cv2.resize(prob_map, (w, h))
            mask_map = cv2.inRange(prob_map, thresh, 1.0)
            contours, _ = cv2.findContours(mask_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                mask_cnt = np.zeros(mask_map.shape)
                mask_cnt = cv2.fillConvexPoly(mask_cnt, contour, 1)
                prob_cnt = prob_map * mask_cnt
                _, _, _, max_loc = cv2.minMaxLoc(prob_cnt)
                point = max_loc + (prob_map[max_loc[1], max_loc[0]], auto_id)
                one_part_points.append(point)
                auto_id += 1
                point_list.append(point)
            all_points.append(one_part_points)
        point_list = np.array(point_list)
        # for i, points in enumerate(all_points):
        #     print(self.points[i], points)

        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        valid_pairs = []
        invalid_pairs = []
        for k in range(len(self.PAFs)):
            paf_a = output[0, self.PAFs[k][0], :, :]
            paf_b = output[0, self.PAFs[k][1], :, :]
            paf_a = cv2.resize(paf_a, (w, h))
            paf_b = cv2.resize(paf_b, (w, h))

            points_a = all_points[self.pairs[k][0]]
            points_b = all_points[self.pairs[k][1]]

            if len(points_a) != 0 and len(points_b) != 0:
                valid_pair = np.zeros((0, 3))
                for i in range(len(points_a)):
                    max_j = -1
                    max_score = -1
                    found = False
                    for j in range(len(points_b)):
                        d_ij = np.subtract(points_b[j][:2], points_a[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue

                        interp_coord = list(zip(np.linspace(points_a[i][0], points_b[j][0], num=n_interp_samples),
                                                np.linspace(points_a[i][1], points_b[j][1], num=n_interp_samples)))

                        paf_interp = []
                        for m in range(len(interp_coord)):
                            paf_interp.append([paf_a[int(round(interp_coord[m][1])), int(round(interp_coord[m][0]))],
                                               paf_b[int(round(interp_coord[m][1])), int(round(interp_coord[m][0]))]])

                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)

                        if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > max_score:
                                max_j = j
                                max_score = avg_paf_score
                                found = True
                    if found:
                        valid_pair = np.append(valid_pair, [[points_a[i][3], points_b[max_j][3], max_score]], axis=0)
                valid_pairs.append(valid_pair)
            else:
                # print("No connection: k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])

        person_points = -1 * np.ones((0, 19))
        for k in range(len(self.PAFs)):
            if k not in invalid_pairs:
                points_a = valid_pairs[k][:, 0]
                points_b = valid_pairs[k][:, 1]
                index_a, index_b = np.array(self.pairs[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(person_points)):
                        if person_points[j][index_a] == points_a[i]:
                            person_idx = j
                            found = 1
                            break
                    if found:
                        person_points[person_idx][index_b] = points_b[i]
                        person_points[person_idx][-1] += point_list[points_b[i].astype(int)][2] + valid_pairs[k][i][2]
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[index_a] = points_a[i]
                        row[index_b] = points_b[i]
                        row[-1] = sum(point_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        person_points = np.vstack([person_points, row])

        final_points = []
        for i, person in enumerate(person_points):
            if person[-1] < 5:
                continue
            points = []
            for j, part in enumerate(person[:-1]):
                if part == -1:
                    points.append((-1, -1))
                    continue
                x = np.int32(point_list[int(part)][0])
                y = np.int32(point_list[int(part)][1])
                points.append((x, y))
            final_points.append(points)
        final_points = np.array(final_points)

        return final_points

    def draw(self, image, one_person_points, thickness=4):
        for i, pair in enumerate(self.pairs):
            x1 = one_person_points[pair[0]][0]
            y1 = one_person_points[pair[0]][1]
            x2 = one_person_points[pair[1]][0]
            y2 = one_person_points[pair[1]][1]
            if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1:
                continue
            cv2.line(image, (x1, y1), (x2, y2), self.colors[i], thickness)

if __name__ == "__main__":
    _pose = OpenPose()

    file = open("dataset_example/data.csv", "r")

    _dir = "dataset/pose_only_pose/"
    lines = file.readlines()

    file.close()

    points_data = []


    for line in lines:
        parts = line.split(" ")
        name = parts[0]
        data = parts[1:37]
        points = []

        for i in range(18):
            x = int(data[i * 2])
            y = int(data[i * 2 + 1])
            points.append((x, y))
        points_data.append(points)


    for k in range(len(points_data)):
        print('points_data', points_data[k])

        array_x = []
        array_y = []

        for i in range(18):
            num_x = points_data[k][i][0]
            num_y = points_data[k][i][1]

            if num_x and num_y > 0:
                array_x.append(num_x)
                array_y.append(num_y)

        max_x = max(array_x)
        min_x = min(array_x)
        max_y = max(array_y)
        min_y = min(array_y)
        #print(max_x, min_x, max_y, min_y)

        new_points_data = []

        for j in range(18):
            num_x = points_data[k][j][0]
            num_y = points_data[k][j][1]

            if num_x and num_y > 0:
                num_x = num_x - min_x
                num_y = num_y - min_y
                new_points_data.append((num_x, num_y))
            else:
                num_x = -1
                num_y = -1
                new_points_data.append((num_x, num_y))

        height = max_y - min_y
        width = max_x - min_x

        image = np.zeros((height,width,3), np.uint8)

        new_image = np.zeros((100, 100, 3), np.uint8)
        _pose.draw(image, new_points_data)

        img = str(int(time.time())) + ".jpg"

        # n_w = 100
        # n_h = 100
        #
        # if image.shape[1] > image.shape[0]:
        #     n_h = int(image.shape[0] * n_w / image.shape[1])
        # else:
        #     n_w = int(image.shape[1] * n_h / image.shape[0])
        #
        # ox = int((100 - n_w) / 2)
        # oy = int((100 - n_h) / 2)
        #
        # print(oy, oy+n_h, ox, ox+n_w)
        #
        # print(image.shape)
        #
        # image = cv2.resize(image, (n_w, n_h))
        #
        # new_image[oy:n_h + oy, ox:n_w + ox, :] = image[0:image.shape[0], 0:image.shape[1], :]
        #
        # cv2.imwrite(_dir + img, new_image)
        #
        # cv2.imshow('image', new_image)
        # cv2.waitKey(100)

        cv2.imwrite(_dir + img, image)

        cv2.imshow('image', image)
        cv2.waitKey(100)