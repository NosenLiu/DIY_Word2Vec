#conding:utf-8

"""使用jieba分词，将语料文件进行逐次分割。并产生一个 字典，存储常用6000词，存入json文件。"""

import numpy as np
import jieba 
import json
import time, threading
#from multiprocessing import process   # 多进程进行
import multiprocessing      # 主进程，子进程用 multiprocessing.Manager().list() 等传数据
from multiprocessing import Process,Value,Array
from multiprocessing import Pool


# lock = threading.Lock()
"""服务于8个进程"""
# TODO 全局变量 TODO      
pured_file = '../dataset/news_sohusite_xml_pure_200M_lines_content.txt'  # 未切分语料地址

# TODO 全局变量 TODO

# array_c,array_di,array_dn 是三个返回变量，multiprocessing.Manager().list 类型
def cut_content_to_dict(c_lines,proc_index,array_c,array_di,array_dn):   
    print("process %s, length %s"%(str(proc_index),str(len(c_lines))))
    for sentences in c_lines:
        words = jieba.cut(sentences,cut_all=False)
        words = ' '.join(words)  # 使用 空格 将每一个词汇隔开 此时是str类型变量。
        array_c.append(words[:-1])   # 去除句末的 '\n'
    count = 0
    for item in array_c:
        words = item.split(' ')
        for wd in words:
            if wd in array_di:
                array_dn[array_di.index(wd)] += 1   # 有这个单词，则对应计数加1
            else:
                array_di.append(wd)
                array_dn.append(1)
        if count%250==0:
            print("process %s, finished %s"%(str(proc_index),str(count)))
        count+=1
    # finish_tag[proc_index] = 1
    # print(finish_tag)
    # if 0 not in finish_tag:   # finish_tag 全为1 时，启用拼接函数
    #     put_together()

def put_together(corp_list_all,dict_list_all,dict_list_index_all):
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
# TODO 测试 4000行数据
        # lines = lines[:4000]

        block_size = int(len(lines)/8)
        tasks_8 = []
        for i in range(7):
            tasks_8.append(lines[i*block_size:(i+1)*block_size])
        tasks_8.append(lines[7*block_size:])

        proc_id = []
        corp_list_all = []     # jieba分词过后的语料文件
        dict_list_all = []    # 用于每个进程的词汇列表
        dict_list_index_all = []  #用于每个进程的 词汇列表对于的出现次数。
        #finish_tag = [0,0,0,0,0,0,0,0]
        for i in range(8):
            corp_list_all.append(multiprocessing.Manager().list([]))   #  主进程与子进程的共享list
        for i in range(8):
            dict_list_all.append(multiprocessing.Manager().list([]))   #  主进程与子进程的共享list
        for i in range(8):
            dict_list_index_all.append(multiprocessing.Manager().list([]))   #  主进程与子进程的共享list
        for i in range(8):
            proc_id.append(multiprocessing.Value("i",i))   # "i" 表示int类型， 第二个i表示初始数值

        p_list = []
        for i in range(8):
            p = multiprocessing.Process(target=cut_content_to_dict,args=(tasks_8[i],proc_id[i],\
                corp_list_all[i],dict_list_all[i],dict_list_index_all[i]))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        
        put_together(corp_list_all,dict_list_all,dict_list_index_all)


        # p = Pool()
        # for i in range(8):
        #     p.apply_async(cut_content_to_dict, args=(tasks_8[i],proc_id[i],\
        #         corp_list_all[i],dict_list_all[i],dict_list_index_all[i]))
        # p.close()
        # p.join()
        print('外层主进程已完结？')

        
    
        
    
    
