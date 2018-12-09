# DIY_Word2Vec
Use skipgram to build word2vec.

2018/12/09 Update
使用搜狐新闻语料库数据
URL:  http://www.sogou.com/labs/resource/cs.php
语料预处理工作。
进行汉语分词、统计工作。由于上传大小受限，仅上传迷你示例数据news_sohusite_xml.smarty_1.dat。
split_words.py 对原始.dat文件进行处理，提取新闻内容存入XXXX_pure_content.txt。之后，使用jieba分词，对新闻内容进行分词，之后统计词汇出现频率，对出现频率高的前10000个词汇进行收录，按出现频率由高到低的顺序以字典形式存储在XXXX_pure_content_dict.json文件中，同时存储分词后的新闻语料XXXX_pure_content_jieba.txt做后续训练使用。（三段jieba_XXX.py代码的功能相同，分别使用多线程、多进程进行实现）
