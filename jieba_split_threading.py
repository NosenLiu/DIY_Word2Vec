#conding:utf-8

"""使用jieba分词，将语料文件进行逐次分割。并产生一个 字典，存储常用6000词，存入json文件。"""

import numpy as np
import jieba 
import json
import time, threading  # 用多线程方式加速运行

# def open_file_to_cut(path):
#     splited_list = []
#     with open(path,'r',encoding='utf-8') as fin:
#         lines = fin.readlines()
#         for sentences in lines:
#             words = jieba.cut(sentences,cut_all=False)
#             words = ' '.join(words)  # 使用 空格 将每一个词汇隔开 此时是str类型变量。
#             splited_list.append(words[:-1])   # 去除句末的 '\n'
#     word_index_list = []
#     word_num_list = []
#     f_corp = open(path[:-4]+'_jieba.txt','w',encoding='utf-8')  # TODO 语料文件
#     for item in splited_list:
#         f_corp.write(item+'\n')
#         words = item.split(' ')
#         for wd in words:
#             if wd in word_index_list:
#                 word_num_list[word_index_list.index(wd)] += 1   # 有这个单词，则对应计数加1
#             else:
#                 word_index_list.append(wd)
#                 word_num_list.append(1)
#     f_corp.close()
#     word_num_list = np.array(word_num_list)
#     word_frequent_list = np.argsort(-word_num_list)    # 降序排序，并获取其排序的索引顺序
#     d_out = dict()            # TODO 最终常用词词典列表
#     count = 0
#     for index in word_frequent_list[:6000]:
#         d_out[word_index_list[index]] = count
#         count += 1
#     #print(d_out['不能'])
#     with open(path[:-4]+'_dict.json','w',encoding='utf-8') as f_dict:
#         json.dump(d_out, f_dict)


"""服务于8个线程"""
# TODO 全局变量 TODO      
pured_file = '../dataset/news_sohusite_xml_pure——200万行_content.txt'  # 未切分语料地址
corp_list_all = []     # jieba分词过后的语料文件
for i in range(8):
    corp_list_all.append([])
dict_list_all = []    # 用于每个进程的词汇列表
for i in range(8):
    dict_list_all.append([])
dict_list_index_all = []  #用于每个进程的 词汇列表对于的出现次数。
for i in range(8):
    dict_list_index_all.append([])
finish_tag = [0,0,0,0,0,0,0,0]
# TODO 全局变量 TODO

def cut_content_to_dict(c_lines,thread_index):
    print("thread %s, length %s"%(str(thread_index),str(len(c_lines))))
    splited_list = []
    for sentences in c_lines:
        words = jieba.cut(sentences,cut_all=False)
        words = ' '.join(words)  # 使用 空格 将每一个词汇隔开 此时是str类型变量。
        splited_list.append(words[:-1])   # 去除句末的 '\n'
    corp_list_all[thread_index] = splited_list    # 将分词结果存入全局变量
    word_index_list = []     #临时存储词汇列表
    word_num_list = []       #临时存储词汇出现次数
    count = 0
    for item in splited_list:
        words = item.split(' ')
        for wd in words:
            if wd in word_index_list:
                word_num_list[word_index_list.index(wd)] += 1   # 有这个单词，则对应计数加1
            else:
                word_index_list.append(wd)
                word_num_list.append(1)
        if count%250==0:
            print("thread %s, finished %s"%(str(thread_index),str(count)))
        count+=1
    dict_list_all[thread_index] = word_index_list       #将词汇表存入全局变量对应位置
    dict_list_index_all[thread_index] = word_num_list   #将词汇频率存入全局变量对应位置
    finish_tag[thread_index] = 1
    if 0 not in finish_tag:   # finish_tag 全为1 时，启用拼接函数
        put_together()

def put_together():
    print("22222",time.time())
    for i in range(8)[1:]:
        # 对每一个词汇表进行遍历，同 dict_list_all[0] 进行比较，如果词汇相同则数目累加，如无该词汇则填充。
        for j in range(len(dict_list_all)):
            if dict_list_all[i][j] in dict_list_all[0]:
                dict_list_index_all[0][dict_list_all[0].index(dict_list_all[i][j])] += dict_list_index_all[i][j]
            else:
                dict_list_all[0].append(dict_list_all[i][j])
                dict_list_index_all[0].append(dict_list_index_all[i][j])
    dict_list_index_last = np.array(dict_list_index_all[0])
    word_frequent_list = np.argsort(-dict_list_index_last)    # 降序排序，并获取其排序的索引顺序
    d_out = dict()            # TODO 最终常用词词典列表
    count = 0
    for index in word_frequent_list[:10000]:    # 提取10000个常用词汇
        d_out[dict_list_all[0][index]] = count
        count += 1
    with open(pured_file[:-4]+'_jieba.txt','w',encoding='utf-8') as f_corp:
        for i in corp_list_all:             # 将分词后语料写入txt文档。
            for j in i:
                f_corp.write(j+'\n')
    with open(pured_file[:-4]+'_dict.json','w',encoding='utf-8') as f_dict:
        json.dump(d_out, f_dict)
    print("33333",time.time())



if(__name__=="__main__"):
    print("11111",time.time())
    with open(pured_file,'r',encoding='utf-8') as content_file:
        lines = content_file.readlines()

        lines = lines[:4000]

        block_size = int(len(lines)/8)
        tasks_8 = []
        for i in range(7):
            tasks_8.append(lines[i*block_size:(i+1)*block_size])
        tasks_8.append(lines[7*block_size:])
        thread_list = []
        for i in range(8):
            thread_list.append(threading.Thread(target=cut_content_to_dict, args=(tasks_8[i],i)))
        for i in range(8):
            thread_list[i].start()
    
        
    
    
