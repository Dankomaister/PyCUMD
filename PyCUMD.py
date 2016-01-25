'''
Created on 16 jan. 2016

@author: danhe
'''

import numpy as np
from math import ceil
from pycuda import autoinit, driver, compiler, gpuarray
from gpu_prefixsum import DoubleBuffer

def read_xyz(file_name):
    
    i = 0
    ion_type = []
    coordinate = ''
    
    file = open(file_name, 'r')
    lines = file.readlines()
    ions = int(lines[0])
    
    while i < ions:
        line = lines[2 + i].split()
        
        ion_type.append(line[0])
        coordinate += line[1] + ' ' + line[2] + ' ' + line[3] + '\n'
        
        i += 1
    
    coordinate = np.fromstring(coordinate, sep=' ').reshape(ions, 3)
    
    return ions, ion_type, coordinate

class Simulation(object):
    '''
    classdocs
    '''
    
    def __init__(self, coordinate, mass, box_size, cutoff, eps, sig, dt=1):
        '''
        Constructor
        '''
        
        autoinit.context.set_cache_config(driver.func_cache.PREFER_L1)
        
        # Scalars
        self.dfloat = 'float32'
        self.dint = 'int32'
        self.iter = np.int32(0)
        self.kb = np.float32(8.6173324e-5)
        
        self.ions = np.int32(coordinate.shape[0])
        self.mass = np.float32(mass)
        self.cutoff = np.float32(cutoff)
        
        self.eps = np.float32(eps)
        self.sig = np.float32(sig)
        self.dt = np.float32(dt)
        
        self.threads = 256
        self.block = (self.threads, 1, 1)
        self.grid = (ceil(self.ions / self.threads), 1, 1)
        
        self.ion_type = ['Ar'] * self.ions
        
        # float3
        self.box_size = np.array(box_size).astype(self.dfloat)
        
        # 1D arrays
        self.potential_energy = gpuarray.zeros(self.ions, np.float32)
        self.kinetic_energy = gpuarray.zeros(self.ions, np.float32)
        
        # 3D arrays
        self.coordinate = gpuarray.to_gpu_async(coordinate.astype(self.dfloat))
        self.coordinate_sorted = gpuarray.to_gpu_async(coordinate.astype(self.dfloat))
        self.velocity = gpuarray.zeros_like(self.coordinate)
        self.velocity_sorted = gpuarray.zeros_like(self.coordinate)
        self.force = gpuarray.zeros_like(self.coordinate)
        
        # Timers
        self.start = driver.Event()
        self.end = driver.Event()
        self.timer = 0
        
        # System data
        self.system_pe = np.array([]).astype(self.dfloat)
        self.system_ke = np.array([]).astype(self.dfloat)
        
        # Create the bins
        self.create_bins()
        
        # Load kernels
        float3 = gpuarray.vec.float3
        int3 = gpuarray.vec.int3
        
        self.prefixsum = DoubleBuffer(self.bins, self.threads)
        self.fill_bins = self.load_function('lj_force.cu', 'fillBins', ('PPP', float3, int3, 'i'))
        self.counting_sort = self.load_function('lj_force.cu', 'countingSort', 'PPPPPPi')
        self.lj_force = self.load_function('lj_force.cu', 'ljForce', ('PPPPP', float3, float3, int3, 'fffi'))
        self.verlet_pre = self.load_function('lj_force.cu', 'verletPre', ('PPPPP', float3, 'ffi'))
        self.verlet_pos = self.load_function('lj_force.cu', 'verletPos', 'PPPffi')
        
        return
    
    def read_kernel(self, file_name):
        
        file = open(file_name, 'r')
        kernel = ''.join(file.readlines())
        
        return kernel
    
    def load_function(self, file_name, func_name, func_input):
        
        mod = compiler.SourceModule(self.read_kernel(file_name), options=['-use_fast_math'], no_extern_c=True)
        # mod = compiler.SourceModule(self.read_kernel(file_name), options=['-use_fast_math', '--maxrregcount=32'], no_extern_c=True)
        
        func = mod.get_function(func_name)
        func.prepare(func_input)
        
        return func
    
    def create_bins(self):
        
        self.bin_dim = np.floor(2 * self.box_size / self.cutoff).astype(self.dint)
        self.bin_length = (self.box_size / self.bin_dim).astype(self.dfloat)
        self.bins = np.prod(self.bin_dim)
        
        self.bin_index = gpuarray.zeros(self.ions, np.int32)
        self.bin_count = gpuarray.zeros(self.bins, np.int32)
        
        return
    
    def write_xyz(self, file_name, write_delay=1, io='w'):
        
        if self.iter % write_delay == 0:
            
            coordinate = self.coordinate_sorted.get()
            
            file = open(file_name, io)
            
            file.write('%i\n' % self.ions)
            file.write('Current iteration %i, simulation time %f fs.\n' % (self.iter, self.iter * self.dt))
            
            for i in zip(self.ion_type, coordinate[:, 0], coordinate[:, 1], coordinate[:, 2]):
                file.write('%3s %8.3f %8.3f %8.3f\n' % i)
            
            file.close()
            
            return
        else:
            return
    
    def get_force(self):
        
        self.fill_bins.prepared_call(self.grid, self.block,
                                     self.coordinate.gpudata, self.bin_index.gpudata, self.bin_count.gpudata,
                                     self.bin_length, self.bin_dim, self.ions)
        
        self.prefixsum.run(self.bin_count)
        
        self.counting_sort.prepared_call(self.grid, self.block,
                                         self.bin_index.gpudata, self.prefixsum.output.gpudata, self.coordinate.gpudata, self.velocity.gpudata,
                                         self.coordinate_sorted.gpudata, self.velocity_sorted.gpudata, self.ions)
        
        self.lj_force.prepared_call(self.grid, self.block,
                                    self.coordinate_sorted.gpudata, self.force.gpudata,
                                    self.potential_energy.gpudata, self.bin_count.gpudata, self.prefixsum.output.gpudata,
                                    self.box_size, self.bin_length, self.bin_dim, self.cutoff, self.eps, self.sig, self.ions)
        
        self.bin_count = gpuarray.zeros(self.bins, np.int32)
        
        return
    
    def verlet_part1(self):
        
        self.verlet_pre.prepared_call(self.grid, self.block,
                                      self.coordinate.gpudata, self.velocity.gpudata,
                                      self.coordinate_sorted.gpudata, self.velocity_sorted.gpudata,
                                      self.force.gpudata, self.box_size, self.mass, self.dt, self.ions)
        
        return
    
    def verlet_part2(self):
        
        self.verlet_pos.prepared_call(self.grid, self.block,
                                      self.velocity_sorted.gpudata, self.force.gpudata, self.kinetic_energy.gpudata,
                                      self.mass, self.dt, self.ions)
        
        return
    
    def run_nve(self):
        
        self.start.record()
        self.start.synchronize()
        
        self.verlet_part1()
        self.get_force()
        self.verlet_part2()
        
        self.end.record()
        self.end.synchronize()
        
        self.timer += 0.001 * self.start.time_till(self.end)
        
        self.iter += 1
        
        return
    
    def get_system_info(self):
        
        self.system_pe = np.append(self.system_pe, np.sum(self.potential_energy.get()))
        self.system_ke = np.append(self.system_ke, np.sum(self.kinetic_energy.get()))
        
        return
    
    def print_status(self, write_delay=1):
        
        if self.iter < 2:
            print('PyCUMD')
            print('   Simulation box size: (%1.2f %1.2f %1.2f)' % (self.box_size[0], self.box_size[1], self.box_size[2]))
            print('   %i atoms' % self.ions)
            print('   %i bins' % self.bins)
            print('   %i threads\n' % self.threads)
            print('%9s %13s %13s %13s %13s' % ('Iter', 'Kin. E. (eV)', 'Pot. E. (eV)', 'Tot. E. (eV)', 'Perf. (Mp/s)'))
        
        elif self.iter % write_delay == 0:
            
            self.get_system_info()
            
            print('%9i %13.5f %13.5f %13.5f %13.5f' % (self.iter, self.system_ke[-1], self.system_pe[-1],
                                                     self.system_ke[-1] + self.system_pe[-1],
                                                     1e-6 * self.ions * self.iter / self.timer))
            
            return
        else:
            return
