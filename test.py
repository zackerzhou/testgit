#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time
import csv
from itertools import combinations_with_replacement
import timeit
import torch
import gc

sys.path.append('/pytorch-cifar-master/')

# mode = 'one'
mode = 'two'

# one_exec = 'cuda'

# co_exec = 'cuda'
co_exec = 'nn'

path_exe = ['PolyBench_exe', 'pytorch-cifar-master']
poly_exe = ['2DConvolution', '3DConvolution', '3mm', 'atax', 'bicg', 'gemm', 'gesummv', 'mvt', 'syr2k', 'syrk',
            'fdtd2d', 'correlation', 'covariance']
# poly_exe = ['2mm','gramschmidt']
epoch = '32'

inference_exe = [(epoch, '1', 'test', 'VGG'), (epoch, '1', 'test', 'ResNet'), (epoch, '1', 'test', 'GoogleNet'), (epoch, '1', 'test', 'DenseNet'), (epoch, '1', 'test', 'MobileNet'),
                 (epoch, '16', 'test', 'VGG'), (epoch, '16', 'test', 'ResNet'), (epoch, '16', 'test', 'GoogleNet'), (epoch, '16', 'test', 'DenseNet'), (epoch, '16', 'test', 'MobileNet'),
                 (epoch, '32', 'test', 'VGG'), (epoch, '32', 'test', 'ResNet'), (epoch, '32', 'test', 'GoogleNet'),(epoch, '32', 'test', 'DenseNet'), (epoch, '32', 'test', 'MobileNet'),
                 (epoch, '64', 'test', 'VGG'), (epoch, '64', 'test', 'ResNet'), (epoch, '64', 'test', 'GoogleNet'), (epoch, '64', 'test', 'DenseNet'), (epoch, '64', 'test', 'MobileNet')]



one_task_time_csv = 'one_task_time.csv'
two_task_time_csv = 'two_task_time.csv'

threshold = 50      # if the execution time of any benchmark is less than the threshold, the benchmark will be iteratively executed

def execute_one():


    # ***** execute CUDA *****
    bench = 'PolyBench_exe'
    for i in poly_exe:
        loop_time, t = 0, 0
        while(t < threshold):
            # ****** time: start *******
            start_time = time.time()
            print('Loop: %d, Executing of *** %s *** in %s' % (loop_time, i, bench))
            p1 = subprocess.Popen(
                './%s/%s.exe' % (bench, i), close_fds=True,
                shell=True, preexec_fn=os.setsid)

            while 1:
                ret = subprocess.Popen.poll(p1)
                if ret == 0:
                    break
                elif ret is None:
                    pass

            end_time = time.time()
            exec_time = (end_time - start_time)
            print('execution time of the task is %f' % exec_time)

            loop_time += 1          # number of loops
            t += exec_time          # total execution of the benchmark (multiple loops)


        print('Job: %s, Exec: %d, total time: %f seconds' % (i, loop_time, t))
        f = open(one_task_time_csv, 'a+')
        csv_w = csv.writer(f)
        csv_w.writerow([i, t, loop_time])
        time.sleep(2)
        f.close()


    bench = 'pytorch-cifar-master'
    for i in inference_exe:
        loop_time, t = 0, 0
        str_i = '-'
        while(t < threshold):
            # ****** time: start *******
            start_time = time.time()
            print('Loop:%d, Executing of *** %s ***, epoch:%s,batch:%s,mode:%s' % (loop_time, i[3], i[0], i[1], i[2]))
            p1 = subprocess.Popen(
                'python ./%s/main_arg.py  --epoch %s --batch %s --job %s --net %s' % (bench, i[0], i[1], i[2], i[3]), close_fds=True,
                shell=True, preexec_fn=os.setsid)


            while 1:
                ret = subprocess.Popen.poll(p1)
                if ret == 0:
                    break
                elif ret is None:
                    pass

            end_time = time.time()
            exec_time = (end_time - start_time)

            loop_time += 1
            t += exec_time

        print('Job: %s, Exec: %d, total time: %f seconds' % ("i", loop_time, t))
        f = open(one_task_time_csv,'a+')
        csv_w = csv.writer(f)
        csv_w.writerow([str_i.join(i), t, loop_time])
        time.sleep(2)
        f.close()


def execute_two():
    # read the loop time and execute time from "one_task_time.csv"
    # format ---> [name of benchmark, execution time, loop time]
    dict_one = {}
    f = open(one_task_time_csv, 'r')
    csv_r = csv.reader(f)
    for p in csv_r:
        name = p[0]
        dict_one[name] = p  # {'name':['name', exec_time, loop_time]}
    f.close()

    # emerge the inference_exe's name
    # ('32','64','test','MobileNet') ---> '32-64-test-MobileNet'
    str_i = '-'
    inference_exe_join = []
    for i in inference_exe:
        inference_exe_join.append(str_i.join(i))

    # poly_exe + inference_exe_join
    # ['2DConvolution', '3DConvolution', ...] + ['32-1-test-VGG', '32-16-test-VGG', ...]
    # em_exe = poly_exe + inference_exe_join


    em_exe = poly_exe + inference_exe
    # execute two tasks in order
    for p in combinations_with_replacement(em_exe,2):
        i, j = '2DConvolution', '3mm'
        print('Execute *** %s *** & *** %s ***'% (i, j))
        c_1, c_2 = 0, 0     # 0: PolyBench_exe, 1: inference_exe

        # verify the category of the tasks
        if i in poly_exe:
            print('task 1 in poly_exe')
            c_1 = 0
            n_1, t_1, l_1 = dict_one[i]  # obtain the name, exectuion time, loop time of task 1
        elif i in inference_exe:
            print('task 1 in inference_exe')
            c_1 = 1
            str_temp = '-'
            n_1, t_1, l_1 = dict_one[str_temp.join(i)]
        else:
            n_1, t_1, l_1 = 0, 0, 0
            print('task 1 do not belong to any category')
            exit(0)

        l_1 = int(l_1)

        if j in poly_exe:
            print('task 2 in poly_exe')
            c_2 = 0
            n_2, t_2, l_2 = dict_one[j]  # obtain the name, exectuion time, loop time of task 1
        elif j in inference_exe:
            print('task 2 in inference_exe')
            c_2 = 1
            str_temp = '-'
            n_2, t_2, l_2 = dict_one[str_temp.join(j)]
        else:
            n_2, t_2, l_2 = 0, 0, 0
            print('task 2 do not belong to any category')
            exit(0)

        l_2 = int(l_2)


        cnt_1, cnt_2 = 0, 0
        total_exec_1, total_exec_2 = 0, 0
        s_1, s_2 = 0, 0     # 0: task stopped, 1: task executing
        o_1, o_2 = 0, 0     # 0: loop, 1: loop over
        test_1, test_2 = 0, 0
        while(o_1 == 0 or o_2 == 0):
            print('loop of task 1 and 2: %d & %d' % (cnt_1, cnt_2))
            print('o_1 and o_2: %d & %d' % (o_1, o_2))
            # execute task 1
            if o_1 == 0 and s_1 == 0:
                test_1 += 1
                s_1 = 1     # task 1 is being executed
                if c_1 == 0:
                    print('task 1 is in poly_exe')
                    bench1 = 'PolyBench_exe'
                    start_time_1 = time.time()
                    p1 = subprocess.Popen(
                        './%s/%s.exe' % (bench1, n_1),close_fds=True,
                        shell=True, preexec_fn=os.setsid)
                elif c_1 == 1:
                    print('task 1 is in inference_exe')
                    bench1 = 'pytorch-cifar-master'
                    start_time_1 = time.time()
                    p1 = subprocess.Popen(
                        'python ./%s/main_arg.py --epoch %s --batch %s --job %s --net %s' % (bench1, i[0], i[1], i[2], i[3]),close_fds=True,
                        shell=True, preexec_fn=os.setsid)
                else:
                    print('c_1 error, exit')
                    exit(0)
            else:
                pass

            # execute task 2
            if o_2 == 0 and s_2 == 0:
                test_2 += 1
                s_2 = 1     # task 2 is being executed
                if c_2 == 0:
                    print('task 2 is in poly_exe')
                    bench2 = 'PolyBench_exe'
                    start_time_2 = time.time()
                    p2 = subprocess.Popen(
                        './%s/%s.exe' % (bench2, n_2),close_fds=True,
                        shell=True, preexec_fn=os.setsid)
                elif c_2 == 1:
                    print('task 2 is in inference_exe')
                    bench2 = 'pytorch-cifar-master'
                    start_time_2 = time.time()
                    p2 = subprocess.Popen(
                        'python ./%s/main_arg.py --epoch %s --batch %s --job %s --net %s' % (
                        bench2, j[0], j[1], j[2], j[3]),close_fds=True,
                        shell=True, preexec_fn=os.setsid)
                else:
                    print('c_2 error, exit')
                    exit(0)
            else:
                pass

            # control loop
            # e_1, e_2 = 0, 0
            while 1:
                ret1 = subprocess.Popen.poll(p1)
                ret2 = subprocess.Popen.poll(p2)

                # if ret1 == 0 and ret2 == 0 and s_1 == 1 and s_2 == 1:
                #     print('two tasks are stopped at the same time')
                #     print(n_1,n_2)
                #     exit(0)

                # print(s_1,s_2)
                # print(ret1,ret2)
                # print('ret1:%d, ret2:%d' %(ret1,ret2))
                if ret1 == 0 and s_1 == 1:
                    # if ret1 == 0 and ret2 == 0:
                    #     print('task 1 is stopped')
                    #     print(ret1,ret2)
                    #     print('s_1:%d,s2:%d' % (s_1,s_2))
                    #     print('c_1:%d,c_2:%d' % (c_1,c_2))
                    cnt_1 += 1
                    s_1 = 0
                    # e_1 = 1
                    end_time_1 = time.time()
                    exec_time_1 = end_time_1 - start_time_1
                    print('Execution time of task 1 is %f' % exec_time_1)       # ********
                    total_exec_1 += exec_time_1
                    o_1 = int(cnt_1 >= l_1)
                    # if ret1 == 0 and ret2 == 0:
                    #     test = input('input:')
                    break
                else:
                    pass

                if ret2 == 0 and s_2 == 1:
                    # if ret1 == 0 and ret2 == 0:
                    #     print('task 2 is stopped')
                    #     print(ret1, ret2)
                    #     print('s_1:%d,s2:%d' % (s_1, s_2))
                    #     print('c_1:%d,c_2:%d' % (c_1, c_2))
                    cnt_2 += 1
                    s_2 = 0
                    # e_2 = 1
                    end_time_2 = time.time()
                    exec_time_2 = end_time_2 - start_time_2
                    print('Execution time of task 2 is %f' % exec_time_2)       # ********
                    total_exec_2 += exec_time_2
                    o_2 = int(cnt_2 >= l_2)
                    # if ret1 == 0 and ret2 == 0:
                    #     test = input('input:')
                    break
                else:
                    pass

                # if cnt_1 >= l_1:
                #     o_1 = 1
                #
                # if cnt_2 >= l_2:
                #     o_2 = 1
                #
                # if e_1 == 1 or e_2 == 1:
                #     break


        # f = open(two_task_time_csv, 'a+')
        # csv_w = csv.writer(f)
        # csv_w.writerow([n_1, n_2, t_1, total_exec_1, t_2, total_exec_2])    # [name of task1, name of task2, execution time of task1(one), execution time of task1(two), execution time of task2(one), execution time of task2(two)]
        # f.close()
        time.sleep(2)
        # torch.cuda.empty_cache()
        print('test: %d & %d' % (test_1, test_2))
        print('cnt: %d & %d' % (cnt_1, cnt_2))

    # exit(0)
    # execute two tasks from CUDA (e.g., PolyBench)
    # if co_exec == 'cuda':
    #     bench = 'PolyBench_exe'
    #     for p in combinations_with_replacement(poly_exe,2):
    #         i, j = p[0], p[1]
    #         print(' *** Executing %s and %s *** ' % (i,j))
    #         s1, s2 = 0, 0  # flag of two jobs (1: completed, 0: ongoing)
    #         exec_time_1, exec_time_2 = 0, 0
    #
    #         start_time = time.time()
    #         p1 = subprocess.Popen(
    #             './%s/%s.exe' % (bench, i),
    #             shell=True, preexec_fn=os.setsid)
    #         p2 = subprocess.Popen(
    #             './%s/%s.exe' % (bench, j),
    #             shell=True, preexec_fn=os.setsid)
    #
    #         while 1:
    #             ret1 = subprocess.Popen.poll(p1)
    #             ret2 = subprocess.Popen.poll(p2)
    #             if ret1 == 0 and s1 == 0:
    #                 s1 = 1
    #                 end_time_1 = time.time()
    #                 exec_time_1 = (end_time_1 - start_time)
    #
    #             if ret2 == 0 and s2 == 0:
    #                 s2 = 1
    #                 end_time_2 = time.time()
    #                 exec_time_2 = (end_time_2 - start_time)
    #
    #             if s1 == 1 and s2 == 1:
    #                 f = open(two_task_time_csv, 'a+')
    #                 csv_w = csv.writer(f)
    #                 csv_w.writerow([i, j, exec_time_1, exec_time_2])
    #                 time.sleep(2)
    #                 f.close()
    #                 break
    #             else:
    #                 pass
    #
    # elif co_exec == 'nn':
    #     # execute two inference tasks
    #     bench = 'pytorch-cifar-master'
    #     for p in combinations_with_replacement(inference_exe,2):
    #         i, j = p[0], p[1]
    #         if i[3] == j[3]:
    #             continue
    #         print(' *** Executing %s and %s *** ' % (i[3],j[3]))
    #
    #         s1, s2 = 0, 0  # flag of two jobs (1: completed, 0: ongoing)
    #         exec_time_1, exec_time_2 = 0, 0
    #
    #         start_time = time.time()
    #         p1 = subprocess.Popen(
    #             'python ./%s/main_arg.py --epoch %s --batch %s --job %s --net %s' % (bench, i[0], i[1], i[2], i[3]),
    #             shell=True, preexec_fn=os.setsid)
    #         p2 = subprocess.Popen(
    #             'python ./%s/main_arg.py --epoch %s --batch %s --job %s --net %s' % (bench, j[0], j[1], j[2], j[3]),
    #             shell=True, preexec_fn=os.setsid)
    #
    #         while 1:
    #             ret1 = subprocess.Popen.poll(p1)
    #             ret2 = subprocess.Popen.poll(p2)
    #             if ret1 == 0 and s1 == 0:
    #                 s1 = 1
    #                 end_time_1 = time.time()
    #                 exec_time_1 = (end_time_1 - start_time)
    #
    #             if ret2 == 0 and s2 == 0:
    #                 s2 = 1
    #                 end_time_2 = time.time()
    #                 exec_time_2 = (end_time_2 - start_time)
    #
    #             if s1 == 1 and s2 == 1:
    #                 f = open(two_task_time_csv, 'a+')
    #                 csv_w = csv.writer(f)
    #                 csv_w.writerow([i, j, exec_time_1, exec_time_2])
    #                 time.sleep(2)
    #                 f.close()
    #                 torch.cuda.empty_cache()
    #                 break
    #             else:
    #                 pass



if __name__ == '__main__':
    # if mode == 'one':
    #     execute_one()
    # elif mode == 'two':
    #     execute_two()



    # bench1 = 'PolyBench_exe'
    # bench2 = 'PolyBench_exe'
    # n_1 = '2DConvolution'
    # n_2 = 'atax'
    # start_time_1 = time.time()
    # p1 = subprocess.Popen(
    #     './%s/%s.exe' % (bench1, n_1), close_fds=True,
    #     shell=True, preexec_fn=os.setsid)
    # start_time_2 = time.time()
    # p2 = subprocess.Popen(
    #     './%s/%s.exe' % (bench2, n_2), close_fds=True,
    #     shell=True, preexec_fn=os.setsid)
    #
    # s1, s2 = 0, 0
    # while 1:
    #     ret1 = subprocess.Popen.poll(p1)
    #     ret2 = subprocess.Popen.poll(p2)
    #
    #
    #     if ret1 == 0 and s1 == 0:
    #         s1 = 1
    #         end_time_1 = time.time()
    #         exec_time_1 = end_time_1 - start_time_1
    #         # print('Execution time of task 1 is %f' % exec_time_1)  # ********
    #
    #     if ret2 == 0 and s2 == 0:
    #         s2 = 1
    #         end_time_2 = time.time()
    #         exec_time_2 = end_time_2 - start_time_2
    #         # print('Execution time of task 2 is %f' % exec_time_2)  # ********
    #
    #     if s1 == 1 and s2 == 1:
    #         break
    #
    # print('Execution time of task 1 is %f' % exec_time_1)  # ********
    # print('Execution time of task 2 is %f' % exec_time_2)  # ********
    # print('two tasks are completed')


    DEVNULL = open(os.devnull, 'wb')


    bench1 = 'PolyBench_exe'
    bench2 = 'PolyBench_exe'
    n_1 = '3DConvolution'
    n_2 = '32-16-test-MobileNet'
    cnt_1, cnt_2 = 0, 0
    s_1, s_2 = 0, 0
    o_1, o_2 = 0, 0
    l_1, l_2 = 22, 5
    sel = 0
    total_exec_1, total_exec_2 = 0, 0
    while(cnt_1 < l_1 or cnt_2 < l_2):
        print('loop of task 1 and 2: %d & %d' % (cnt_1, cnt_2))
        print('o_1 and o_2: %d & %d' % (o_1, o_2))

        if cnt_1 < l_1 and s_1 == 0:
            s_1 = 1
            start_time_1 = time.time()
            p1 = subprocess.Popen(
                './%s/%s.exe' % (bench1, n_1), close_fds=True,
                shell=True, preexec_fn=os.setsid,stdout=DEVNULL,stderr=subprocess.STDOUT)

        if cnt_2 < l_2 and s_2 == 0:
            s_2 = 1
            start_time_2 = time.time()
            p2 = subprocess.Popen(
                './%s/%s.exe' % (bench2, n_2), close_fds=True,
                shell=True, preexec_fn=os.setsid,stdout=DEVNULL,stderr=subprocess.STDOUT)


        while 1:
            ret1 = subprocess.Popen.poll(p1)
            ret2 = subprocess.Popen.poll(p2)

            if ret1 == 0 and s_1 == 1:
                s_1 = 0
                cnt_1 += 1
                end_time_1 = time.time()
                exec_time_1 = end_time_1 - start_time_1
                total_exec_1 += exec_time_1
                print('Execution time of task 1 is %f' % exec_time_1)  # ********
                # o_1 = int(cnt_1 >= l_1)
                break

            if ret2 == 0 and s_2 == 1:
                s_2 = 0
                cnt_2 += 1
                end_time_2 = time.time()
                exec_time_2 = end_time_2 - start_time_2
                total_exec_2 += exec_time_2
                print('Execution time of task 2 is %f' % exec_time_2)  # ********
                # o_2 = int(cnt_2 >= l_2)
                break



    print('two tasks are completed')
    print(total_exec_1,total_exec_2)

    # p1 = subprocess.Popen(
    #             'python test_2D.py', close_fds=True,
    #             shell=True, preexec_fn=os.setsid)
    # p2 = subprocess.Popen(
    #     'python test_3mm.py', close_fds=True,
    #     shell=True, preexec_fn=os.setsid)

