#coding:utf-8

"""使用CBOW(continuous bag of words)方法，构建、训练词向量，其样本模式为多个词输入，单个词汇输出
20181214修改，逐批次生成样本并训练，一次处理20行语料，并训练。否则会内存溢出
20181216修改，一次处理5行语料，避免内存溢出。训练速度不会因此降低"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   #程序运行时不提示warnings
os.environ['CUDA_VISIBLE_DEVICES']='0' 

import numpy as np
import tensorflow as tf

import json
import datetime


class CBOW_Cell(object):
    def __init__(self, window_length=5, word_dim=300):
        with tf.variable_scope("matrix_scope") as matrix_scope:
            self.tar_weight = tf.get_variable(name='tar_weight',shape=[10000,word_dim],\
                initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
            self.front_weight = tf.get_variable(name='front_weight',shape=[1,2*window_length],\
                initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
            self.back_weight = tf.get_variable(name='back_weight',shape=[word_dim,10000],\
                initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
            matrix_scope.reuse_variables()
        # 上方为tar_weight,front_weight,back_weight 三个权重矩阵的维度设置及初始化。
        # 下方为偏移量权重的设置 及 变量保存。
        self.bias = tf.Variable(tf.zeros([1,10000])) # 偏移量，用于加到softmax前的输出上
        self.word_dim = word_dim   # 词向量维度
        self.window_length = window_length    
        # 下方为占位符，规定好输入、输出的shape
        self.sample_in = tf.placeholder(tf.float32, [2*window_length, 10000],name='sample_in')
        self.sample_out = tf.placeholder(tf.float32, [1, 10000],name='sample_out')

    def forward_prop(self,s_input):
        step_one = tf.matmul(s_input,self.tar_weight)
        out_vector = tf.matmul(tf.matmul(self.front_weight,step_one),self.back_weight)+self.bias
        return out_vector
    
    def loss_func(self,lr=0.001):
        out_vector = self.forward_prop(self.sample_in)
        y_pre = tf.nn.softmax(out_vector,name='y_pre')
        # 无负采样loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.sample_out,logits=y_pre)
        # 加负采样loss
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(tf.nn.nce_loss(\
        #     weights=self.tar_weight,biases=self.bias,labels=self.sample_out,\
        #     inputs=tf.transpose(self.sample_in),num_sampled=5,num_classes=10000))

        # mse = tf.sqrt(tf.reduce_mean(tf.square(y_pre-self.batch_out)),name='mse')
        # cross_entropy = -tf.reduce_mean(self.batch_out * tf.log(y_pre),name='cross_entropy')
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
        return y_pre,cross_entropy,train_op

    def train_model(self, savepath,crop_lines_all,index_to_word,word_to_index,epochs=1000,lr=0.001):
        # if 'session' in locals() and session is not None:  # TODO 这个if判断是用来看系统中是否有session，有的话将其关闭
        #     print('Close interactive session')
        #     session.close()
        y_pre,cross_entropy,train_op = self.loss_func(lr)  # TODO TODO TODO  这句话千万不能放到循环里面，会重复绘制计算图！！！运行很慢！！
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for data_num in range(int(len(crop_lines_all)/5)):
                # 生成 in_list out_list 
                in_list,out_list = make_samples(crop_lines_all,index_to_word,\
                    word_to_index,self.window_length,data_num)   #一次20个行的处理语料样本
                out_list = out_list.reshape(len(in_list),1,10000)
                if (data_num)%50==0:
                    print('样本已处理',data_num*5,'/',len(crop_lines_all),'行。 ',datetime.datetime.now().strftime('%H:%M:%S.%f'))
                for i in range(epochs):
                    for j in range(len(in_list)):
                        sess.run(train_op, feed_dict={self.sample_in:in_list[j], \
                            self.sample_out:out_list[j]})
                        # if (j+1)%2000==0:
                        #     print('epoch %d finished %d samples of %d.'%(i,j+1,len(in_list)),sess.run(cross_entropy,feed_dict={\
                        #         self.sample_in:in_list[j], self.sample_out:out_list[j]}),' ',datetime.datetime.now().strftime('%H:%M:%S.%f'))
        #下面为存储模型的代码
            tar_weight=self.tar_weight.eval()   # 这个就是词向量表[10000*词向量维度]，是word2vec的最终目标
            front_weight=self.front_weight.eval()
            back_weight=self.back_weight.eval()
            bias=self.bias.eval()
            word_dim=self.word_dim
            window_length=self.window_length
            np.savez(savepath,tar_weight=tar_weight,front_weight=front_weight,\
                back_weight=back_weight,bias=bias,word_dim=word_dim,window_length=window_length)
            print('model saved in:',savepath)

    # retrain 函数，多一个 modelpath参数，用于区别载入模型和在训练后的存储模型
    def retrain(self,modelpath,savepath,crop_lines_all,index_to_word,word_to_index,epochs=1000,lr=0.001):
        # if 'session' in locals() and session is not None:  # TODO 这个if判断是用来看系统中是否有session，有的话将其关闭
        #     print('Close interactive session')
        #     session.close()
        y_pre,cross_entropy,train_op = self.loss_func(lr)  # TODO TODO TODO  这句话千万不能放到循环里面，会重复绘制计算图！！！运行很慢！！
        with tf.Session() as sess:
            update = self.set_params(modelpath)
            sess.run(update)
            for data_num in range(int(len(crop_lines_all)/5)):
                # 生成 in_list out_list 
                in_list,out_list = make_samples(crop_lines_all,index_to_word,\
                    word_to_index,self.window_length,data_num)   #一次20个行的处理语料样本
                out_list = out_list.reshape(len(in_list),1,10000)
                if (data_num)%50==0:
                    print('样本已处理',data_num*5,'/',len(crop_lines_all),'行。 ',datetime.datetime.now().strftime('%H:%M:%S.%f'))
                for i in range(epochs):
                    for j in range(len(in_list)):
                        sess.run(train_op, feed_dict={self.sample_in:in_list[j], \
                            self.sample_out:out_list[j]})
                        # if (j+1)%2000==0:
                        #     print('epoch %d finished %d samples of %d.'%(i,j+1,len(in_list)),sess.run(cross_entropy,feed_dict={\
                        #         self.sample_in:in_list[j], self.sample_out:out_list[j]}),' ',datetime.datetime.now().strftime('%H:%M:%S.%f'))
        #下面为存储模型的代码
            tar_weight=self.tar_weight.eval()   # 这个就是词向量表[10000*词向量维度]，是word2vec的最终目标
            front_weight=self.front_weight.eval()
            back_weight=self.back_weight.eval()
            bias=self.bias.eval()
            word_dim=self.word_dim
            window_length=self.window_length
            np.savez(savepath,tar_weight=tar_weight,front_weight=front_weight,\
                back_weight=back_weight,bias=bias,word_dim=word_dim,window_length=window_length)
            print('model saved in:',savepath)

    def set_params(self,filepath):   # 用于加载现有模型参数的赋值操作
        param_dict = np.load(filepath)
        tar_weight = param_dict['tar_weight']
        front_weight = param_dict['front_weight']
        back_weight = param_dict['back_weight']
        bias = param_dict['bias']
        word_dim = param_dict['word_dim']
        window_length = param_dict['window_length']
        #开始赋值
        update = []
        update.append(tf.assign(self.tar_weight,tar_weight))   # assign是赋值操作，需要在session下run才可以赋值
        update.append(tf.assign(self.front_weight,front_weight))
        update.append(tf.assign(self.back_weight,back_weight))
        update.append(tf.assign(self.bias,bias))
        self.word_dim = word_dim
        self.window_length = window_length
        return update
        # 返回update 操作，在外部的session中再运行，避免反复initialize.
        # with tf.Session() as sess:
        #     sess.run(update)
        # print('successfully load the model ',filepath)
    
# TODO 神经网络类编辑完成  TODO





# 读字典函数
def read_dict(dict_path):
    with open(dict_path,'r') as load_f:
        load_dict = json.load(load_f)    # 此时load_dict为 dict类 变量，可以使用词快速找出其index
    #初始化词列表,用于使用index快速找出其对应的词。
    word_list = list(np.zeros(10000))
    for key in load_dict.keys():
        word_list[load_dict[key]]=key
    return word_list,load_dict #返回 索引→词(list) 和 词→索引(dict) 

# 根据一列词汇来计算输入 one-hot 矩阵
def input_matrix_calc(word_to_index,word_list):
    temp_matrix = []
    for wd in word_list:
        index = word_to_index[wd]
        temp_vector = np.zeros(10000)
        temp_vector[index] = 1.0
        temp_matrix.append(temp_vector)
    return_matrix = np.array(temp_matrix)
    return return_matrix

def make_samples(crop_lines_all,index_to_word,word_to_index,window_len,i):   #参数中的i指第几轮语料
    # 一次处理5行语料防止内存溢出
    crop_lines = crop_lines_all[i*5:(i+1)*5]
    sample_in_list = []     # 输入样本list
    sample_out_list = []    # 输出样本list
    for line in crop_lines:
        line_list = line.split(' ')
        line_list = [word for word in line_list if word in index_to_word]
        if len(line_list)<window_len*2+1:     # 如果语句词汇过少，则抛弃这条语句
            continue
        else:
            # 词语大于双倍窗口的情况下，可以开始拼接样本
            for i2 in range(len(line_list)):
                # 句子开头几个词语，前侧的词语数量不够window_len，则后侧多取一些词语攒齐2*window_len的长度
                if i2<window_len+1:   
                    temp_line_list = line_list[:i2]+line_list[i2+1:2*window_len+1]
                    sample_in_list.append(input_matrix_calc(word_to_index,temp_line_list))
                    temp_out_sample = np.zeros(10000)
                    temp_out_sample[word_to_index[line_list[i2]]] = 1.0
                    sample_out_list.append(temp_out_sample)
                # 句子末尾几个词语，后侧的词语数量不够window_len，则前侧多取一些词语攒齐2*window_len的长度
                elif i2>=len(line_list)-window_len: 
                    temp_line_list = line_list[len(line_list)-2*window_len-1:i2]+line_list[i2+1:]
                    sample_in_list.append(input_matrix_calc(word_to_index,temp_line_list))
                    temp_out_sample = np.zeros(10000)
                    temp_out_sample[word_to_index[line_list[i2]]] = 1.0
                    sample_out_list.append(temp_out_sample)
                # 处于中间阶段，前窗口和后窗口都不越界
                else:
                    temp_line_list = line_list[i2-window_len:i2]+line_list[i2+1:i2+1+window_len]
                    sample_in_list.append(input_matrix_calc(word_to_index,temp_line_list))
                    temp_out_sample = np.zeros(10000)
                    temp_out_sample[word_to_index[line_list[i2]]] = 1.0
                    sample_out_list.append(temp_out_sample)
    return np.array(sample_in_list),np.array(sample_out_list)
    

# 根据训练结果.npz文件，获取词向量表
def get_word_vector(filepath):
    param_dict = np.load(filepath)
    tar_weight = param_dict['tar_weight']
    return tar_weight

def find_nearest_words(tar_weight,wrod,index_to_word,word_to_index):
    dist_vec = []
    for i in range(len(tar_weight)):
        dist = np.linalg.norm(tar_weight[word_to_index[wrod]] - tar_weight[i])
        dist_vec.append(dist)
    dist_closest = np.argsort(dist_vec)
    for index in dist_closest[:10]:
        print(index_to_word[index],dist_vec[index])


if __name__ == "__main__":
    # TODO 参数区 TODO
    window_len = 7
    lr = 0.001
    word_dim = 300
    dict_path = '../dataset/news_sohusite_xml_pure_200M_lines_content_dict.json'  # 分词字典所在路径
    crop_path = '../dataset/news_sohusite_xml_pure_200M_lines_content_jieba.txt'  # 语料文件路径，jieba分词后
    save_path = crop_path[:-4]+'_CBOW.npz'       # 保存npz文件的路径
    # TODO 参数区 TODO
    index_to_word,word_to_index = read_dict(dict_path)

    #  下方 训练区域，如已有训练结果.npz文件，可以将这几行注释掉。 TODO
    with open(crop_path,'r',encoding='utf-8') as crop_file:
        crop_lines_all = crop_file.readlines()
    # 首次训练
    # crop_lines_all = crop_lines_all[:4000]    # 平均1000行一个小时，语料共38万行。
    # cbow_obj = CBOW_Cell(window_length=window_len,word_dim=word_dim)
    # cbow_obj.train_model(save_path,crop_lines_all,index_to_word,word_to_index,epochs=1,lr=0.005)
    # 再次训练
    # crop_lines_all = crop_lines_all[120000:130000]    # 平均1000行一个小时，语料共38万行。
    # cbow_obj = CBOW_Cell(window_length=window_len,word_dim=word_dim)
    # cbow_obj.retrain(save_path[:-4]+'120000.npz',save_path[:-4]+'130000.npz',crop_lines_all,index_to_word,word_to_index,epochs=1,lr=10.5)
    #  上方 训练区域，如已有训练结果.npz文件，可以将这几行注释掉。 TODO



    #  下方 根据训练结果获取 词向量表。  TODO
    # w2v = get_word_vector(save_path)
    w2v = get_word_vector(save_path[:-4]+'120000.npz')
    print(type(w2v),'!!!!!',w2v.shape)
    
    dist = np.linalg.norm(w2v[word_to_index['车辆']] - w2v[word_to_index['车子']]) 
    print('\"车辆\" 与 \"车子\" 之间的欧式距离为：',dist,'!!')
    dist = np.linalg.norm(w2v[word_to_index['机械']] - w2v[word_to_index['工业化']]) 
    print('\"机械\" 与 \"工业化\" 之间的欧式距离为：',dist,'!!')
    dist = np.linalg.norm(w2v[word_to_index['车辆']] - w2v[word_to_index['茶叶']]) 
    print('\"车辆\" 与 \"茶叶\" 之间的欧式距离为：',dist,'!!')
    dist = np.linalg.norm(w2v[word_to_index['粮食']] - w2v[word_to_index['手表']]) 
    print('\"粮食\" 与 \"手表\" 之间的欧式距离为：',dist,'!!')
    
    #  上方 根据训练结果获取 词向量表。  TODO






