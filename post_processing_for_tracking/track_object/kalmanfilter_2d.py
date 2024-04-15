from post_processing_for_tracking.track_object.utils.kalmanfilter_init import KalmanFilter
import numpy as np
import time


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, results=None, check_frame_head=1):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = bbox[5]
        self.class_id = bbox[4]
        self.center_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        self.check_direction = []
        self.direction = 2
        self.class_head_p2 = None
        self.count_head_p2 = 0
        self.center_bbox_motor = (bbox[1] + bbox[3]) / 2
        self.bbox_motor = bbox
        self.bbox_head_driver = []
        # attribute
        humans = results.humans
        heads = results.heads
        self.is_P2 = False
        self.is_P0 = False
        self.check_frame_head = check_frame_head
        self.human_bboxes = []
        self.bbox_head = []
        self.head_h = []
        for human in humans:
            box = human.get_box_info()
            heads1 = human.heads
            if len(heads1) == 1:
                self.head_h.append(1)
            else:
                self.head_h.append(0)
            self.human_bboxes.append([box[0], box[1], box[2], box[3], box[4], box[5], float(box[6])])
        for head in heads:
            box = head.get_box_info()
            self.bbox_head.append([box[0], box[1], box[2], box[3], box[4], box[5], float(box[6])])

    def overlap_ratio(sefl, boxA, boxB):
        """Calculate iou of 2 bboxes

        Args:
            boxA (list): [x1, y1, x2, y2]
            boxB (list): [x1, y1, x2, y2]

        Returns:
            overlap_ratio (float): overlap ratio max(interArea / float(boxBArea), interArea / float(boxAArea))
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        if boxBArea == 0 or boxAArea == 0:
            return 0
        overlap_ratio = max(interArea / float(boxBArea), interArea / float(boxAArea))
        # return the overlap ratio
        return overlap_ratio

    def direct_detection(self, center_point_current):
        """Detect the motor direction 1 if IN detection and 0 if OUT detection

        Args:
            center_point_current (list): motor center coordinates
        """
        if self.check_direction is not None:
            if center_point_current[1] - self.center_point[1] > 0:
                self.check_direction.append(1)
            else:
                self.check_direction.append(0)
            if len(self.check_direction) >= 3:
                rs_in = self.check_direction.count(1)
                rs_out = self.check_direction.count(0)
                if rs_in >= rs_out:
                    self.direction = 1  # in
                else:
                    self.direction = 0  # out
                self.check_direction = None
        self.center_point = center_point_current

    def P2_checking(self):
        """Checking if P2 is on the motorbike
        """
        if self.class_head_p2 is None:
            if len(self.bbox_head) == 3:  # > 2
                self.count_head_p2 += 1
            if self.count_head_p2 > self.check_frame_head:  # 1
                self.class_head_p2 = 'P2'

    def Detecting_P02(self):
        """Reassigning new class for humans on the motorbike
        """
        if self.class_head_p2 == "P2":  # Detecting P2
            c = 0
            bbox_head_check = []
            for head in self.bbox_head:
                center_y = (head[1] + head[3]) / 2
                if center_y < self.center_bbox_motor:
                    bbox_head_check.append(head)
                    c += 1
            if c == 3:
                # self.is_P2 = True
                bbox_human_driver = []
                for human in self.human_bboxes:
                    if human[-2] in [1, 2]:
                        bbox_human_driver.append(human)
                if len(bbox_human_driver) > 0:
                    bbox_human_driver_sort = np.argmax(np.array(bbox_human_driver)[:, -1])
                    bbox_human_driver = bbox_human_driver[bbox_human_driver_sort]
                    bbox_centerX_human_driver = (bbox_human_driver[2] + bbox_human_driver[0]) / 2
                    bbox_Y_human_driver = bbox_human_driver[1]
                    distance_XY_head_driver = []
                    for head in bbox_head_check:
                        bbox_centerX_head = (head[2] + head[0]) / 2
                        bbox_Y_head = head[1]
                        distance_XY_head_driver.append(
                            abs(bbox_centerX_human_driver - bbox_centerX_head) + abs(bbox_Y_human_driver - bbox_Y_head))
                    check_xy_head_driver = np.argmin(np.array(distance_XY_head_driver))
                    bbox_head_driver = bbox_head_check[check_xy_head_driver]
                    bbox_head_check.pop(check_xy_head_driver)
                    center_y_head_driver = (bbox_head_driver[1] + bbox_head_driver[3]) / 2
                    check_head_P2 = False
                    for head in bbox_head_check:
                        center_y_head = (head[1] + head[3]) / 2
                        if self.direction == 1:
                            if center_y_head > center_y_head_driver:
                                check_head_P2 = True
                        elif self.check_direction == 0:
                            if center_y_head < center_y_head_driver:
                                check_head_P2 = True
                    if check_head_P2 is False:
                        self.is_P2 = True
        if self.direction == 1:  # Detecting P0
            c = 0
            bbox_head_check = []
            bbox_human_driver = []
            if len(self.bbox_head) > 1:
                for head in self.bbox_head:
                    top_head = head[0]
                    bottom_head = head[2]
                    center_head_y = (head[1] + head[3]) / 2
                    for human in self.human_bboxes:
                        if human[-2] in [1, 2]:
                            iou = self.overlap_ratio(head, human)
                            top_driver = human[0]
                            left_driver = human[1]
                            bottom_driver = human[2]
                            if iou > 0.99 and center_head_y < self.center_bbox_motor and top_head > top_driver and left_driver < center_head_y and bottom_head < bottom_driver:
                                c += 1
                                bbox_head_check.append(head)
                                bbox_human_driver.append(human)
                                break
            if c > 1:
                bbox_human_driver_sort = np.argmax(np.array(bbox_human_driver)[:, -1])
                bbox_human_driver = bbox_human_driver[bbox_human_driver_sort]
                bbox_centerX_human_driver = (bbox_human_driver[2] + bbox_human_driver[0]) / 2
                bbox_Y_human_driver = bbox_human_driver[1]
                distance_XY_head_driver = []
                for head in bbox_head_check:
                    bbox_centerX_head = (head[2] + head[0]) / 2
                    bbox_Y_head = head[1]
                    distance_XY_head_driver.append(
                        abs(bbox_centerX_human_driver - bbox_centerX_head) + abs(bbox_Y_human_driver - bbox_Y_head))
                check_xy_head_driver = np.argmin(np.array(distance_XY_head_driver))
                bbox_head_driver = bbox_head_check[check_xy_head_driver]
                self.bbox_head_driver = bbox_head_driver
                center_head_y_head_driver = (bbox_head_driver[1] + bbox_head_driver[3]) / 2
                bbox_head_check.pop(check_xy_head_driver)
                for head in bbox_head_check:
                    center_head_y = (head[1] + head[3]) / 2
                    iou_head_driver_p0 = self.overlap_ratio(bbox_head_driver, head)
                    h_head = head[3] - head[1]
                    if iou_head_driver_p0 < 0.1 and h_head < abs(
                            center_head_y - center_head_y_head_driver) < 2 * h_head:
                        for human_p0 in self.human_bboxes:
                            if human_p0[-2] in [7, 8]:
                                iou_head_human_P0 = self.overlap_ratio(human_p0, head)
                                if iou_head_human_P0 > 0.99:
                                    self.is_P0 = True

    def update(self, bbox, results):
        """
        Updates the state vector with observed bbox.
        """
        humans = results.humans
        heads = results.heads
        self.bbox_motor = bbox
        self.center_bbox_motor = (bbox[1] + bbox[3]) / 2
        self.human_bboxes = []
        self.bbox_head = []
        for head in heads:
            box = head.get_box_info()
            self.bbox_head.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if len(bbox) > 0:
            self.kf.update(convert_bbox_to_z(bbox))
        self.score = bbox[5]
        self.class_id = bbox[4]
        self.head_h = []
        for human in humans:
            box = human.get_box_info()
            heads1 = human.heads
            if len(heads1) == 1:
                self.head_h.append(1)
            else:
                self.head_h.append(0)
            self.human_bboxes.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])
        # check direction
        current_center_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        self.direct_detection(current_center_point)
        ######## find p2
        self.P2_checking()
        ### Detecting P0 P2
        self.Detecting_P02()

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
