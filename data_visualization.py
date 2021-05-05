#!/usr/bin/env python

import os
import matplotlib.pyplot as plt


def isplit_by_n(ls, n):
    for i in range(0, len(ls), n):
        if i+n >= len(ls):
            yield sum(ls[i:-1])/(len(ls)-i)
        else:
            yield sum(ls[i:i+n])/n

def split_by_n(ls, n):
    return list(isplit_by_n(ls, n))


def plot_bs():
    path = os.getcwd()+'/agnews/bs/'
    file_list = os.listdir(path)
    for file in file_list:
        if file.endswith('.out'):
            f = open(path+file, 'r')
            data = [float(line.rstrip()) for line in f]
            f.close()

            avg_data = split_by_n(data, 50)
            n = len(avg_data)
            y = avg_data
            x = [i / n * 100 for i in range(1, n + 1)]
            plt.plot(x, y, label=file[-6:-4])

    plt.title('Training Loss vs Step by different Batch Size')
    plt.ylabel('Loss')
    plt.xlabel('Step (%)')
    plt.legend()
    plt.ylim(0.25,0.5)
    plt.show()
    
    
def plot_lr():
    path = os.getcwd()+'/agnews/lr/'
    file_list = os.listdir(path)
    for file in file_list:
        if file.endswith('.out'):
            f = open(path+file, 'r')
            data = [float(line.rstrip()) for line in f]
            f.close()

            avg_data = split_by_n(data, 50)
            n = len(avg_data)
            y = avg_data
            x = [i / n * 100 for i in range(1, n + 1)]
            label = file[-7:-4]
            label = label[:2]+'-'+label[2:]
            plt.plot(x, y, label=label)

    plt.title('Training Loss vs Step by different Learning Rate')
    plt.ylabel('Loss')
    plt.xlabel('Step (%)')
    plt.legend()
    plt.ylim(0.2,1)
    plt.show()
    

if __name__ == '__main__':
    plot_bs()
    plot_lr()