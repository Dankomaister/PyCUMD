'''
Created on 26 dec. 2015

@author: danhe
'''

import numpy as np
from math import floor, ceil, log
from pycuda import compiler, gpuarray

class DoubleBuffer(object):
    '''
    classdocs
    '''
    
    def __init__(self, array_size, threads):
        '''
        Constructor
        '''
        
        self.array_size = array_size
        
        self.block = (threads, 1, 1)
        self.grid_size = floor(log(self.array_size, threads)) + 1
        
        self.grid = [None] * self.grid_size
        self.block_sum = [None] * self.grid_size
        
        for i in range(self.grid_size):
            self.grid[i] = (ceil(self.array_size / threads ** (i + 1)), 1, 1)
            self.block_sum[i] = gpuarray.zeros(self.grid[i][0], np.int32)
        
        self.output = gpuarray.zeros(self.array_size, np.int32)
        
        self.block_add = self.load_function('prefix_sum_double_buffer.cu', 'blockAdd', 'PPi')
        self.prefix_sum = self.load_function('prefix_sum_double_buffer.cu', 'prefixSum', 'PPPi')
        
        return
    
    def read_kernel(self, file_name):
        
        file = open(file_name, 'r')
        kernel = ''.join(file.readlines())
        
        return kernel
    
    def load_function(self, file_name, func_name, func_input):
        
        mod = compiler.SourceModule(self.read_kernel(file_name), options=['-use_fast_math'])
        
        func = mod.get_function(func_name)
        func.prepare(func_input)
        
        return func
    
    def run(self, gpu_input):
        
        i = 0
        self.prefix_sum.prepared_call(self.grid[0], self.block,
                                      gpu_input.gpudata, self.output.gpudata, self.block_sum[0].gpudata, self.array_size)
        i += 1
        
        while i < self.grid_size:
            self.prefix_sum.prepared_call(self.grid[i], self.block,
                                          self.block_sum[i - 1].gpudata, self.block_sum[i - 1].gpudata, self.block_sum[i].gpudata, self.array_size)
            i += 1
        
        i = self.grid_size - 2
        
        while i > 0:
            self.block_add.prepared_call(self.grid[i], self.block,
                                         self.block_sum[i - 1].gpudata, self.block_sum[i].gpudata, self.grid[i - 1][0])
            i -= 1
        
        self.block_add.prepared_call(self.grid[0], self.block,
                                     self.output.gpudata, self.block_sum[0].gpudata, self.array_size)
        
        return

class BalancedTrees(object):
    '''
    classdocs
    '''
    
    def __init__(self, array_size, threads):
        '''
        Constructor
        '''
        
        self.array_size = array_size
        
        self.block = (threads, 1, 1)
        self.grid_size = floor(log(self.array_size, threads)) + 1
        
        self.grid = [None] * self.grid_size
        self.block_sum = [None] * self.grid_size
        
        for i in range(self.grid_size):
            self.grid[i] = (ceil(0.5 * self.array_size / threads ** (i + 1)), 1, 1)
            self.block_sum[i] = gpuarray.zeros(self.grid[i][0], np.int32)
        
        self.output = gpuarray.zeros(self.array_size, np.int32)
        
        self.block_add = self.load_function('prefix_sum_balanced_trees.cu', 'blockAdd', 'PPi')
        self.prefix_sum = self.load_function('prefix_sum_balanced_trees.cu', 'prefixSum', 'PPPi')
        
        return
    
    def read_kernel(self, file_name):
        
        file = open(file_name, 'r')
        kernel = ''.join(file.readlines())
        
        return kernel
    
    def load_function(self, file_name, func_name, func_input):
        
        mod = compiler.SourceModule(self.read_kernel(file_name), options=['-use_fast_math'])
        
        func = mod.get_function(func_name)
        func.prepare(func_input)
        
        return func
    
    def run(self, gpu_input):
        
        i = 0
        self.prefix_sum.prepared_call(self.grid[0], self.block,
                                      gpu_input.gpudata, self.output.gpudata, self.block_sum[0].gpudata, self.array_size)
        i += 1
        
        while i < self.grid_size:
            self.prefix_sum.prepared_call(self.grid[i], self.block,
                                          self.block_sum[i - 1].gpudata, self.block_sum[i - 1].gpudata, self.block_sum[i].gpudata, self.array_size)
            i += 1
        
        i = self.grid_size - 2
        
        while i > 0:
            self.block_add.prepared_call(self.grid[i], self.block,
                                         self.block_sum[i - 1].gpudata, self.block_sum[i].gpudata, self.grid[i - 1][0])
            i -= 1
        
        self.block_add.prepared_call(self.grid[0], self.block,
                                     self.output.gpudata, self.block_sum[0].gpudata, self.array_size)
        
        return
