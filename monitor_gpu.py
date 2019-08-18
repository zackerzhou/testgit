#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
from def_para import NUM_SAMPLE,NET
import datetime

# path =  './power_gpu_2s_multi_net'
# path =  './power_gpu_100ms'
# path =  './power_gpu_100ms_multi_gpu'
path =  './power_gpu_100ms_test'
# monitoring the GPU through nvidia-smi (NVML-based library)

# *** test the time for read gpu info once ***
# start_total = time.time()
# loop = 6
# for i in range(loop):
#     p = os.popen("nvidia-smi >> ./%s/res_%s.txt" % (path, NET))
# end_total = time.time()
# exec_time = (end_total-start_total)
# print(exec_time/loop)
# exit()
# start_total = time.time()
# time.sleep(0.01)
# end_total = time.time()
# exec_time = (end_total-start_total)
# print(exec_time)
# exit()
# *****************    end   *****************


# important: When the number of sampling is larger, the command "os.popen("nvidia-smi >> ./%s/res_%s_%d.txt" % (path, NET, i))" takes about 60ms once
i = 1
while i <= NUM_SAMPLE:
    # p = os.popen("nvidia-smi >> ./power_gpu_2s/res_%s_%d.txt" % (NET, i))
    # p=os.popen("nvidia-smi >> ./power_gpu_2s_multi_gpu/res_%s_%d.txt" %(NET,i))
    # p = os.popen("nvidia-smi >> ./power_gpu_2s_multi_net/res_%s_%d.txt" % (NET, i))
    p = os.popen("nvidia-smi >> ./%s/res_%s_%d.txt" % (path, NET, i))
    print(' ***NUM: %d *** ' % i)
    # print(p)
    i = i+1
    time.sleep(0.01)


# test of nvprof
# import torch
# from torch.autograd import Variable
# x = Variable(torch.randn(5,5), requires_grad=True).cuda()
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     y = x**2
#     y.backward()
# print(prof)
# print(x)


# command for generate gpu info (log)
# timeout -t 130 nvidia-smi --query-gpu=fan.gpu,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 0.01 > test.csv