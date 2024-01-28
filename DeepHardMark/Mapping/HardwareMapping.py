#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch

# from flags import parse_handle

# parsing input parameters
# parser = parse_handle()
# args = parser.parse_args()

# settings
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


class HardwareMapping():
	def __init__(self):
		return

	def load_map(self, filepath):
		with open(filepath,'r') as file:
			file.read('test')

	def save_map(self, filepath):
		with open(filepath,'w') as file:
			file.write('test')

	def make_map(self, operations, NBlocks):
		'''
			Trivial mapping should be replaced with a mapping specific to the target hardware.
			Each operation is mapped to a HW block in order.
			target_var assumes shape = (Batch_size, ...)
		'''

		self.NBlocks = NBlocks

		self.HW_Map = {}

		for ops, var in operations.items():

			variable_shape = (var).shape[1:]

			n=1
			for s in variable_shape:
				n = n*s

			var_Map = np.arange(0,n*var.shape[0]).reshape(var.shape) % n
			var_Map = var_Map % self.NBlocks

			self.HW_Map[ops] = torch.from_numpy(var_Map).cuda()

		return self.HW_Map

	def get(self,name):
		var_map = self.HW_Map[name]

		sparse_map = np.zeros((self.NBlocks,)+var_map.shape[1:])

		for i in range(self.NBlocks):
			mapped_to_block_i = var_map == i
			sparse_map[i:i+1][mapped_to_block_i] = 1

		return sparse_map




if __name__ == "__main__":
	image = np.zeros((1, 32, 32, 3))


	MACs = 16*16

	mapping = HardwareMapping()

	operations = {"images":image, "images2":image}
	mapping.make_map(operations,MACs)
	op_map = mapping.get("images")




