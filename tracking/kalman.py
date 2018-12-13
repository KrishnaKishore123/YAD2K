#! /usr/bin/env python3

'''

This file implements kalman filter.

'''

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
	def __init__(self, dim_x=2, dim_z=1, A=None, B=None, H=None):
		self.dim_x = dim_x
		self.dim_z = dim_z
		self.X = np.zeros(dim_x)
		self.A = A
		self.B = B
		self.P = np.zeros(dim_x*dim_x).reshape((dim_x,dim_x))
		self.H = H
		self.R = np.eye(dim_z)
		self.Q = np.eye(self.dim_x)

	def calculate_kalman_gain(self):
		num = np.dot(self.P_pred, self.H.T)
		denom = np.dot(np.dot(self.H,self.P_pred.T),self.H.T) + self.R*np.random.random_sample()
		self.K = np.dot(num,np.linalg.inv(denom))

	def predict(self, u=None):
		self.X_pred = np.dot(self.A,self.X)
		#print(self.X_pred)
		if u is not None:
			self.X_pred += np.dot(self.B,u)
		Q = self.Q*np.random.random_sample()
		self.P_pred = np.dot(self.A,np.dot(self.P,self.A.T)) + Q

	def update(self, z):
		self.calculate_kalman_gain()
		I_KH = np.eye(self.dim_x)-np.dot(self.K,self.H)
		self.P = np.dot(I_KH, self.P_pred)
		err = z-np.dot(self.H,self.X_pred)
		self.X = self.X_pred + np.dot(self.K,err)
		#print(self.X)

if __name__ == "__main__":
	A = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
	H = np.array([[1,0,0,0],[0,1,0,0]])
	kf = KalmanFilter(dim_x=4,dim_z=2,A=A,H=H)
	x = sorted(np.random.random_sample(20)*10)
	y = x + np.random.random_sample(20)*2
	for i in range(len(x)):
		plt.scatter(x[i],y[i]+10)
		kf.predict()
		kf.update(np.array([x[i],y[i]+10]))
		plt.plot(kf.X[0],kf.X[1],c='r')
	plt.show()
