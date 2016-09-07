"""
Author: CHANDRAMOHAN T N
File: Utils_parallel.py 
"""

import os
import multiprocessing
import itertools
import numpy as np

def Create_data(features, seq, lab, o_f):
    l = len(features)
    count = 0
    data = []
    data_dict = {}
    batch_size = 1000
    for d in seq:
	p = multiprocessing.Pool()
	args = []
	if d not in data_dict:
            for i in range(l):
                args.append((d, features[i]))
            v = p.map(Count_subsequence, args)
            p.close()
            p.join()
        else:
            v = data_dict[d]

        data.append(v)
	if len(data) == batch_size:
            data = np.asarray(data)
            filename = o_f + str(count)
            n_lab = lab[(count * batch_size): ((count + 1) * batch_size)]
            n_lab = np.reshape(n_lab, (len(n_lab), 1))
            np.savez_compressed(filename, data=data, lab=n_lab)
            print(count)
            count += 1
            data = []

    if len(data) > 0:
        data = np.asarray(data)
        filename = o_f + str(count)
        n_lab = lab[(count * batch_size): ]
        n_lab = np.reshape(n_lab, (len(n_lab), 1))
        np.savez_compressed(filename, data=data, lab=n_lab)
        print(count)
    print('Dataset created .........')

def Count_subsequence(args):
    S, s = args
    m, n = len(S), len(s)
    table = [0] * n
    for i in xrange(m):
        prev = 1
        for j in xrange(n):
            curr = table[j]
            if S[i] == s[j]:
                table[j] += prev
            prev = curr
    return table[n - 1] if n else 1

def Get_data(i_f):
    f = open(i_f, 'r')
    data = []
    while 1:
        line = f.readline()
        line = line[0:len(line) - 1]
        if len(line) > 0:
            items = line.split(',')
            items = items[1:]
            data.append(items)
        else:
            f.close()
            break
    print('Data created .......')
    return data

def Get_features(feat):
    f = open(feat, 'r')
    features = []
    while 1:
        line = f.readline()
        line = line[0:len(line) - 1]
        if len(line) > 0:
            items = line.split(',')
            features.append(items)
        else:
            f.close()
            break
    print('Features created .......')
    return features

def Get_fs(i_f1, i_f2, o_f):
    f = open(i_f1, 'r')
    g = open(o_f, 'w')
    h = open(i_f2, 'r')

    s1 = []
    while 1:
        line = f.readline()
        line = line[0:len(line) - 1]
        if len(line) > 0:
            items = line.split(' ')
            idx = items.index('SUP:')
            items = items[0:(idx - 1)]
            s = ''
            for i in items:
                if i != '-1':
                    s = s + i + ','
            s = s[0:len(s)-1]
            s1.append(s)
        else:
            f.close()
            break
    s2 = []
    while 1:
        line = h.readline()
        line = line[0:len(line) - 1]
        if len(line) > 0:
            items = line.split(' ')
            idx = items.index('SUP:')
            items = items[0:(idx - 1)]
            s = ''
            for i in items:
                if i != '-1':
                    s = s + i + ','
            s = s[0:len(s)-1]
            s2.append(s)
        else:
            h.close()
            break
    #feat = list(set(s1).difference(set(s2))) + list(set(s2).difference(set(s1)))
    feat = list(set(s1) | set(s2))
    for ft in feat:
        g.write(ft + '\n')
    print('Frequent subsequence dataset created ........')

def Get_labels_ids(i_f):
    labels = []
    ids = []
    f = open(i_f, 'r')

    while 1:
        line = f.readline()
        line = line[0:len(line)-1]
        if len(line) > 0:
            items = line.split(',')
            labels.append(items[1])
            ids.append(items[0])
        else:
            f.close()
            break
    print('Read ids and labels .....')
    return labels, ids

def Get_buyers(lab):
    b = []
    nb = []
    for i in range(len(lab)):
        if lab[i] == '1':
            b.append(i)
        else:
            nb.append(i)
    print('Buyers obtained ......')
    return b, nb

def Balance_data(cat, lab, buyers, nbuyers):
    for i in range(3):
        buyers = buyers + buyers
    j = len(nbuyers) - len(buyers)
    buyers = buyers + buyers[0: j]
    l = len(buyers)

    n_cat = []
    for i in range(l):
        n_cat.append(cat[buyers[i]])
        n_cat.append(cat[nbuyers[i]])
    n_lab = []
    for i in range(l):
        n_lab.append(lab[buyers[i]])
        n_lab.append(lab[nbuyers[i]])
    return [n_cat, n_lab]

def main():
    vn = '0.000500'
    vb = '0.00300'
    i_f1 = '../Train_data/Train/Sequence/nb_output_' + vn + '.txt'
    i_f2 = '../Train_data/Train/Sequence/b_output_' + vb + '.txt'
    feat = '../Train_data/Train/Sequence/FSM_features.txt'
    Get_fs(i_f1, i_f2, feat)
    features = Get_features(feat)

    i_f = '../Train_data/Train/Category.dat'
    data = Get_data(i_f)
    i_f = '../Train_data/Train/Labels.dat'
    [lab, ids] = Get_labels_ids(i_f)
    [data, lab] = Balance_data(data, lab)
    o_f = '../Train_data/Train/Sequence/'
    Create_data(features, data, lab, o_f)
    '''
    i_f = '../Train_data/Val/Category.dat'
    data = Get_data(i_f)
    i_f = '../Train_data/Val/Labels.dat'
    [lab, ids] = Get_labels_ids(i_f)
    o_f = '../Train_data/Val/Sequence/'
    Create_data(features, data, lab, o_f)
    
    i_f = '../Train_data/Test/Category.dat'
    data = Get_data(i_f)
    i_f = '../Train_data/Test/Labels.dat'
    [lab, ids] = Get_labels_ids(i_f)
    o_f = '../Train_data/Test/Sequence/'
    Create_data(features, data, lab, o_f)
    '''

if __name__ == "__main__":
    main()
