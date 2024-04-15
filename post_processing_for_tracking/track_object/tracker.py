import numpy as np
from post_processing_for_tracking.track_object.kalmanfilter_2d import KalmanBoxTracker
from post_processing_for_tracking.track_object.utils.data_association import associate_detections_to_trackers


class my_tracking(object):
    def __init__(self, max_age=2, min_hits=0, iou_threshold=0.15, check_frame_head=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.check_frame_head = check_frame_head

    def update(self, dets=np.empty((0, 5)), results=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if len(dets) < 1:
            return [], [], [], []

        self.frame_count += 1
        # # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0], results[d[0]])
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], results[i], self.check_frame_head)
            self.trackers.append(trk)
        i = len(self.trackers)
        ret = []
        ret_P2 = False
        ret_P0 = False
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and trk.hit_streak >= self.min_hits:
                if trk.is_P2:
                    ret_P2 = True
                if trk.is_P0:
                    ret_P0 = True
                if trk.is_P2:
                    ret.append(np.concatenate((d, [trk.class_id], [trk.score], [trk.id], [1])).reshape(1,
                                                                                                       -1))
                else:
                    ret.append(np.concatenate((d, [trk.class_id], [trk.score], [trk.id], [0])).reshape(1,
                                                                                                       -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            ret = np.concatenate(ret)
        return ret, ret_P2, ret_P0
