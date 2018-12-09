#coding:utf-8

"""将编码不统一的文件统一为UTF-8格式，并且提取出<content>标签中的内容"""

import numpy as np
import chardet   # 用于处理编码检测问题
import codecs


def pure_coding(path):
# TODO 将语料文件处理成统一编码     TODO 注意！！！ TODO 这里需要把原始语料文件 .dat 内容重新保存一下、才能正常使用。
    # path = '../dataset/news_sohusite_xml.smarty_short.dat'
    pure_file = open(path[:-4]+"_pure.txt",'w',encoding='utf-8')
    with codecs.open(path,'rb') as f_corp:
        # TODO 去除杂质编码
        count = 0
        count1 = 0
        lines = f_corp.readlines()
        for word in lines:
            try:
                count += 1
                # f_charInfo=chardet.detect(word)
                word_out = word.decode('gbk')
                pure_file.write(word_out)
            except:
                count1 += 1
                code_name = chardet.detect(word)
                if code_name['confidence']>0.90:
                    word_out = word.decode(code_name['encoding'],errors='ignore')
                    pure_file.write(word_out)
                else:
                    continue
    pure_file.close()
    print(count,count1)
# TODO 将语料文件处理成统一编码

def split_content(path):
    # TODO 使用统一编码的文件，提取出新闻内容。
    content_file = open(path[:-4]+"_content.txt",'w',encoding='utf-8')
    with open(path,'r',encoding='utf-8') as f_corp:
        content_lines = f_corp.readlines()
        for item in content_lines:
            if item[:9]=='<content>' and len(item)>20:     # 需判断长度，防止有<content></content> 的情形出现
                content_file.write(item[9:-11]+'\n')
    content_file.close()
    # TODO 使用统一编码的文件，提取出新闻内容。




if __name__ == "__main__":
    path = '../dataset/news_sohusite_whole_xml.dat'
    pure_coding(path)
    split_content(path[:-4]+'_pure.txt')

