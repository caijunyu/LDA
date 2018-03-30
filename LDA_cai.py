# -*- coding: utf-8 -*-
"""
拉取qq聊天记录，用主题模型进行分析
Created on Thu Mar 29 19:37:00 2018

@author: roboCai
"""
import re
import time
import jieba
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
from gensim import corpora, models, similarities



#对信息处理，过滤@某人的情况以及去掉【表情】等。
def clean_info(info):
    replace_str = (('\n', ''), ('\r', ''), (',', '，'), ('表情', ''), ('图片', ''))
    for rs in replace_str:
        info = info.replace(rs[0], rs[1])
        
    at_pattern = re.compile(r'(@.* )')
    at = re.findall(pattern=at_pattern, string=info)
    for a in at:
        info = info.replace(a, '')
    idx = info.find('@')
    if idx != -1:
        info = info[:idx]
    return info
  

def regularize_data(raw_fileName, regularize_output_fileName):
    #正则化提取时间、QQ号、聊天信息
    time_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{1,2}:\d{1,2}:\d{1,2}')
    qq_pattern1 = re.compile(r'([1-9][0-9]{4,})')# QQ号最小是10000
    qq_pattern2 = re.compile(r'(\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*)')
    user_name_pattern1 = re.compile(r'[\u4e00-\u9fa5]{1,}')
        
    f = open(raw_fileName)
    f_output = open(regularize_output_fileName, mode = 'w')
    f_output.write('QQ,Time,Info\n')
    qq = user_name = chat_time = info = ''
    for line in f:
        line = line.strip()
        if line:
            t = re.findall(pattern=time_pattern,string=line)
            qq1 = re.findall(pattern = qq_pattern1, string = line)
            qq2 = re.findall(pattern = qq_pattern2, string = line)
            user_name1 = re.findall(pattern=user_name_pattern1, string=line)
            if (len(t) > 0) and ((len(qq1) > 0) or (len(qq2) > 0)): #如果是用户信息行，读取时间和qq号
                if info: #当上一步读取的Line是新的用户信息是，说明上一次的用户说话完成了，如果info不是空的，说明上一个用户说了有用信息，需要把上一个用户的用户信息与说话内容保存下来，
                    info = clean_info(info) #对用户的信息预处理，去掉表情、图片等。
                    if info: #预处理以后还有信息，再保存下来。
                        info = '%s,%s,%s\n' % (qq, chat_time, info)
                        f_output.write(info)
                        info = ''  #将上一次用户说话的info清空，等待本用户的说话内容写入。
                if len(qq1) > 0:#保存当前用户的qq,chat_time
                    qq = qq1[0]
                if len(qq2) > 0:
                    qq = qq2[0][0]
# =============================================================================
#                 if len(user_name1) > 0:
#                     #print user_name1
#                     user_name = user_name1
# =============================================================================
                chat_time = t[0]
            else:     #如果不是用户信息行，读取用户说的话作为Info.
                info += line
    f.close()
    f_output.close()
    
    
def load_stopWords():
    stopwords = set()
    f = open('stopword.txt')
    for w in f:
        stopwords.add(w.strip().decode('GB18030'))
    f.close
    return stopwords


#分词  将info的信息分词处理
def segment(regularize_output_fileName,regularize_segment_output_fileName):
    stopWords = load_stopWords()
    data = pd.read_csv(regularize_output_fileName,header=0)
    for i, info in enumerate(data['Info']):
        info_words = []
        words = jieba.cut(info)
        for word in words:
            if word not in stopWords:
                info_words.append(word.encode('utf-8'))
        if info_words:
            data.iloc[i,2] = ' '.join(info_words)
        else:
            data.iloc[i,2] = np.nan
    data.dropna(axis = 0, how='any', inplace=True)
    data.to_csv(regularize_segment_output_fileName, sep=',', header=True, index=False)
            

#将同一个人的所有话拼接在一起
def combin(regularize_segment_output_fileName, regularize_segment_combin_output_fileName):
    data = pd.read_csv(regularize_segment_output_fileName, header = 0)
    data['QQ'] = pd.Categorical(data['QQ']).codes
    f_output = open(regularize_segment_combin_output_fileName, mode = 'w')
    f_output.write('QQ,Info\n')
    for qq in data['QQ'].unique():
        info = ' '.join(data[data['QQ'] == qq]['Info'])
        f_output.write('%s,%s\n' % (qq, info))
    f_output.close


def lda(regularize_segment_combin_output_fileName):
    np.set_printoptions(linewidth = 300)
    data = pd.read_csv(regularize_segment_combin_output_fileName, header=0)
    texts = []
    for info in data['Info']:
        texts.append(info.decode('utf-8').split(' '))
    M = len(texts)
    print('文档数目：%d个 \n' % M)
    print '正在建立词典 --'
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print '词典的大小：%d个词\n' % V
    print '正在计算文本向量 --'
    corpus = [dictionary.doc2bow(text) for text in texts]
    print '正在计算文档TF-IDF --'
    t_start = time.time()
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print '建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start)
    print 'LDA模型拟合推断 --'
    num_topics = 20
    t_start = time.time()
    lda = models.LdaModel(corpus = corpus_tfidf, num_topics = num_topics,  id2word=dictionary,
                          alpha = 0.001, eta=0.02, minimum_probability=0,
                          update_every=1, chunksize=100, passes=20)
    print u'LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start)
# =============================================================================
#     # 所有文档的主题
#     doc_topic = [a for a in lda[corpus_tfidf]]
#     print 'Document-Topic:\n'
#     pprint(doc_topic)
# =============================================================================
    #########################      每个主题的词分布        ##############
    num_show_term = 7  # 每个主题显示几个词
    print u'每个主题的词分布：'
    for topic_id in range(num_topics):
        print '主题#%d: \t' % topic_id,
        term_distribute_all = lda.get_topic_terms(topic_id)      
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:,0].astype(np.int)  #term_distribute的第0列是词对应的id，第1列对应的是该词在该主题下的概率
        for t in term_id:
            print dictionary.id2token[t],
        print '\n 概率：\t', term_distribute[:,1]
           
    #########################      随机打印某10个文档的主题       ##############
    num_show_topic = 10  # 每个文档显示前几个主题
    print u'10个用户的主题分布：'
    doc_topics = lda.get_document_topics(corpus_tfidf)  #所有文档的主题分布
    idx = np.arange(M)
    np.random.shuffle(idx)
    idx= idx[:10]
    for i in idx:
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:,1])
        # print topic_distribute
        topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
        print (u'第%d个用户的前%d个主题：' % (i, num_show_topic)), topic_idx
        print topic_distribute[topic_idx]
        
    # 显示这10个文档的主题          画图
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 9), facecolor='w')
    for i, k in enumerate(idx):
        ax = plt.subplot(5, 2, i + 1)
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        ax.stem(topic_distribute, linefmt='g-', markerfmt='ro')
        ax.set_xlim(-1, num_topics + 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel(u"概率")
        ax.set_title(u"用户 {}".format(k))
        ax.grid(b=True)
    plt.xlabel(u"主题", fontsize=14)
    plt.suptitle(u'用户的主题分布', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
        
    
    #########################      计算各个主题的强度      ##################
    print u'\n各个主题的强度:\n'
    topic_all = np.zeros(num_topics)
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    for i in np.arange(M):  # 遍历所有文档
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        topic_all += topic_distribute
    topic_all /= M  # 平均
    idx = topic_all.argsort()
    topic_sort = topic_all[idx]
    print topic_sort
    plt.figure(facecolor='w')
    plt.stem(topic_sort, linefmt='g-', markerfmt='ro')
    plt.xticks(np.arange(idx.size), idx)
    plt.xlabel(u"主题", fontsize=14)
    plt.ylabel(u"主题出现概率", fontsize=14)
    plt.title(u'主题强度', fontsize=18)
    plt.grid(b=True, axis='both')
    plt.show()
        
        


if __name__ == '__main__':
    #一些文件地址
    raw_fileName = u'all_data.txt'
    regularize_output_fileName = u'houser_regularize.csv'
    regularize_segment_output_fileName = u'houser_regularize_segment.csv'
    regularize_segment_combin_output_fileName = u'houser_regularize_segment_combin.csv'
    
# =============================================================================
#     regularize_data(raw_fileName, regularize_output_fileName)
#     segment(regularize_output_fileName, regularize_segment_output_fileName)
#     combin(regularize_segment_output_fileName, regularize_segment_combin_output_fileName)
# =============================================================================
    lda(regularize_segment_combin_output_fileName)
