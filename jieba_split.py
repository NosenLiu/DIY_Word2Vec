#conding:utf-8

"""使用jieba分词，将语料文件进行逐次分割。并产生一个 字典，存储常用6000词，存入json文件。"""

import numpy as np
import jieba 
import json
import time, threading  # 用多线程方式加速运行

def open_file_to_cut(path):
    splited_list = []
    with open(path,'r',encoding='utf-8') as fin:
        lines = fin.readlines()

        lines = lines[:4000]

        count = 0
        for sentences in lines:
            words = jieba.cut(sentences,cut_all=False)
            words = ' '.join(words)  # 使用 空格 将每一个词汇隔开 此时是str类型变量。
            splited_list.append(words[:-1])   # 去除句末的 '\n'
            if(count%1000==0):
                print(count)
            count += 1
    return splited_list

def words_statistics(splited_list,path):
    word_index_list = []
    word_num_list = []
    f_corp = open(path[:-4]+'_jieba.txt','w',encoding='utf-8')  # TODO 语料文件
    count = 0
    for item in splited_list:
        f_corp.write(item+'\n')
        words = item.split(' ')
        for wd in words:
            if wd in word_index_list:
                word_num_list[word_index_list.index(wd)] += 1   # 有这个单词，则对应计数加1
            else:
                word_index_list.append(wd)
                word_num_list.append(1)
        if count%500==0:
            print(count)
        count += 1
    f_corp.close()
    word_num_list = np.array(word_num_list)
    word_frequent_list = np.argsort(-word_num_list)    # 降序排序，并获取其排序的索引顺序
    d_out = dict()            # TODO 最终常用词词典列表
    count = 0
    for index in word_frequent_list[:6000]:
        d_out[word_index_list[index]] = count
        count += 1
    #print(d_out['不能'])
    with open(path[:-4]+'_dict.json','w',encoding='utf-8') as f_dict:
        json.dump(d_out, f_dict)





if(__name__=="__main__"):
    pured_file = '../dataset/news_sohusite_xml_pure——200万行_content.txt'
    t1 = time.time()
    split_result = open_file_to_cut(pured_file)
    words_statistics(split_result,pured_file)
    t2 = time.time()
    print(t2-t1)
