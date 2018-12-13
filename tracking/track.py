#! /usr/bin/env python3

'''

This file implements tracking algorithm based on hungarian assignment and kalman filter.

'''
import numpy as np
from kalman import KalmanFilter
from hungarian import Hungarian

class Tracker(object):
	num_objects = 0
	def __init__(self, init_loc):
		self.X = np.zeros((dim_x,1))
		self.miss_frame_count = 0
		self.object_id = num_objects
		num_objects += 1
		self.X[:4,1] = init_loc
		A = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    	H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
		self.kf = KalmanFilter(dim_x=7,dim_z=4,A=A,H=H)

	def update(self, x):
		self.kf.predict()
		self.kf.update(x)

class TrackAlgo(object):
	def __init__(self, dets):
		self.miss_threshold = 10
		self.trackers = []
		for det in dets:
			self.trackers.append(Tracker(self.bbox_to_x(det)))
		self.matcher = Hungarian()

	def bbox_to_x(self, bbox):
		#converts x,y,w,h to x,y,s,r
		x = bbox[0]
		y = bbox[1]
		w = bbox[2]
		h = bbox[3]
		s = w*h
		r = w/h
		return np.array([x+w/2,y+h/2,s,r]).reshape((4,1))

	def x_to_bbox(self, x):
		#converts x,y,s,r to x,y,w,h
		w = np.sqrt(s*r)
		h = s/w
		return [x-w/2,y-h/2,w,h]

	def calculate_iou(self, tracker, det):
		#calculates iou b/w detection and a tracker
		epsilon = 1e-7
		tb = self.x_to_bbox(tracker.X[:4,1])
		box1 = [tb[0]-tb[2]/2,tb[1]-tb[3]/2,tb[0]+tb[2]/2,tb[1]+tb[3]/2]
		box2 = [det[0]-det[2]/2,det[1]-det[3]/2,det[0]+det[2]/2,det[1]+det[3]/2]
		intersection = max((box1[2]-box2[0])*(box1[3]-box2[1]),0)
		union = tb[2]*tb[3]+det[2]*det[3]-intersection
		return intersection/(union+epsilon)
	
	def generate_iou_matrix(self, trackers, dets):
		mat = np.zeros((len(trackers),len(dets)))
		for i,tracker in enumerate(trackers):
			for j,det in enumerate(dets):
				mat[i,j] = self.calculate_iou(tracker.X[:4,:],det)
		return mat

	def associate_boxes(self,mat):
		return self.matcher.solve(mat)

	def get_loc_estimate(self,tracker,det):
		tracker.update(bbox_to_x(det))
		return self.x_to_bbox(tracker.X[:4,1])

	def update(self, dets):
		mat = self.generate_iou_matrix(self.trackers, dets)
		matches = self.associate_boxes(mat)
		for i,tracker_id in enumerate(matches):
			if mat[tracker_id,i] > min_iou:
				loc = self.get_loc_estimate(trackers[tracker_id],dets[i])
				trackers[tracker_id].miss_frame_count = 0
			else:
				trackers[tracker_id].miss_frame_count += 1
				if trackers[tracker_id].miss_frame_count > self.miss_threshold:
					trackers.pop(tracker_id)
				self.trackers.append(Tracker(self.bbox_to_x(dets[i])))

if __name__=="__main__":
	video_file = "path/to/video/file"
	#TODO: add logic to read video frame by frame and feed to tracking algo
	#process first frame and use dets to init trackers
	learner = TrackAlgo(dets)
	for frame in frames[1:]:
		dets = get_detections(frame)
		learner.update(dets)
		update_display(learner)