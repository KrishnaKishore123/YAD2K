#! /usr/bin/env python3

'''

This file implements hungarian algorithm to solve data association problem.
Association problem is modelled as an assignment problem

'''

import numpy as np

class Hungarian(object):
	def __init__(self):
		pass

	def assign(self, score_matrix):
		out = np.ones(score_matrix.shape[0])*-1
		freeze = False

		while not freeze:
			freeze = True
			for i in range(score_matrix.shape[0]):
				if np.count_nonzero(score_matrix[i,:]==0) > 1 or np.count_nonzero(score_matrix[i,:]==0) == 0:
					continue
				freeze=False
				col_id = np.where(score_matrix[i,:]==0)[0]
				score_matrix[i,col_id] = -2
				score_matrix[:,col_id] = np.where(score_matrix[:,col_id]==0, -1, score_matrix[:,col_id])
				out[i] = col_id

			if np.count_nonzero(out==-1) == 0:
				break

			for i in range(score_matrix.shape[0]):
				if np.count_nonzero(score_matrix[:,i]==0) > 1 or np.count_nonzero(score_matrix[:,i]==0) == 0:
					continue
				freeze=False
				row_id = np.where(score_matrix[:,i]==0)[0]
				score_matrix[row_id, i] = -2
				score_matrix[row_id,:] = np.where(score_matrix[row_id,:]==0, -1, score_matrix[row_id,:])
				out[row_id] = i

		if np.count_nonzero(out==-1) != 0:
			out = []
		return out, score_matrix

	def extract_zero_lines(self,matrix):
		sel_rows = []
		sel_cols = []
		freeze = False
		for row_id in range(matrix.shape[0]):
			if -2 not in matrix[row_id,:]:
				sel_rows.append(row_id)
		while not freeze:
			freeze=True
			for row_id in sel_rows:
				row = np.squeeze(matrix[row_id,:])
				if -1 in row:
					indices = np.argwhere(row==-1)
					for index in indices:
						if index not in sel_cols:
							freeze=False
							sel_cols.append(index)
			for col_id in sel_cols:
				col = np.squeeze(matrix[:,col_id])
				if -2 in col and np.argwhere(col==-2)[0] not in sel_rows:
					freeze=False
					sel_rows.append(np.argwhere(col==-2)[0])

		return sel_rows, sel_cols

	def remake(self, score_matrix, assigned_mat):
		max_element = np.max(score_matrix)
		sel_rows, sel_cols = self.extract_zero_lines(assigned_mat)
				
		temp = score_matrix.copy()
		while True:
			min_loc = np.unravel_index(np.argmin(temp), temp.shape)
			if min_loc[1] in sel_cols or min_loc[0] not in sel_rows:
				temp[min_loc] = max_element+1
			else:
				break
		min_element = score_matrix[min_loc]

		for i in range(score_matrix.shape[0]):
			for j in range(score_matrix.shape[1]):
				if i not in sel_rows and j in sel_cols:
					score_matrix[i,j]+=min_element
				elif i in sel_rows and j not in sel_cols:
					score_matrix[i,j]-=min_element

		return score_matrix

	def solve(self, score_matrix):
		if np.ndim(score_matrix) != 2:
			print("input to hungarian algorithm must be 2 dimensional!!!")
			return []
		
		shape = score_matrix.shape
		if shape[0] < shape[1]:
			padding = np.zeros((shape[1]-shape[0],shape[1]))
			score_matrix = np.vstack((score_matrix,padding))
		elif shape[0] > shape[1]:
			padding = np.zeros((shape[0], shape[0]-shape[1]))
			score_matrix = np.hstack((score_matrix, padding))

		score_matrix = score_matrix.T - score_matrix.min(axis=1)
		score_matrix = score_matrix.T - score_matrix.min(axis=1)
		print(score_matrix)

		res, assigned_mat = self.assign(score_matrix.copy())
		while len(res) == 0:
			#input()
			score_matrix = self.remake(score_matrix, assigned_mat)
			res, assigned_mat = self.assign(score_matrix.copy())

		return res

if __name__=='__main__':
	#matrix = np.array([[1,7,12],[15,10,3],[2,5,8]])
	matrix = np.array([[11,7,10,17,10],[13,21,7,11,13],[13,13,15,13,14],[18,10,13,16,14],[12,8,16,19,10]])
	solver = Hungarian()
	assignment = solver.solve(matrix)
	print(assignment)
