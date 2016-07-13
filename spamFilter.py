# -*- coding:utf8 -*-

import os
import codecs
import re
import json
from collections import Counter

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as n

config_file='./config'
valuestr='value';
spamStr='spam'
hamStr='ham'
#过滤这些字符，可以计数用于逻辑回归
filterLetter_num='1234567890';
filterLetter_punctuation='.-/*+。，“‘’”"\n\'[]\{\}\\=~`~！@#￥%……&*（）;·——+《》？?；<>、|【】,-_!@#$%^&*()：:';
filterLetter_alphabet='qwertyuiopasdfghjkklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM';
probabilty_spam=0.66;#垃圾邮件的概率，垃圾邮件数/邮件总数
probabilty_ham=0.33;#正常邮件的概率，正常邮件数/邮件总数
probabilty_default=0.5;#在字典中找不到的词的默认概率，如果用于生成字典的文件很典型，那么这个值应该高一点。

#由词库文件生成分词字典
def getWorddictByFilename(filename):
    worddict=dict();
    with codecs.open(filename,'r','utf8') as f:
        for line in f:
            line=line.split()
            word=line[0]
            value={'count':line[1]}
            addWordToDict(word,value,worddict);
    return worddict
def initWorddict(filename):
    return getWorddictByFilename(filename)
def loadWorddict(filename):
    return json.load(codecs.open(filename,'r','utf8'))
def saveWorddict(wd,filename):
    json.dump(wd,codecs.open(filename,'w','utf8'))
#向分词字典中添加词语
def addWordToDict(word,value,worddict):    
    currdict=worddict;
    for letter in word:
        if not currdict.get(letter):
            currdict[letter]=dict()
        currdict=currdict.get(letter)
    currdict[valuestr]=value


#使用分词词典worddict切分句子
def splitSentence(sentence,worddict):
    space_count=0
    punctuation_count=0;
    number_count=0;
    alphabet_count=0;
    wordSplitedList=list();
    tempWordList=list();
    lastdict=worddict;
    for letter in sentence:
        #print(letter,tempWordList,lastdict==worddict)
        if letter in ' \t':
            space_count=space_count+1
            continue
        if letter in filterLetter_punctuation:
            punctuation_count=punctuation_count+1;
            continue;
        if letter in filterLetter_num :#and tempWordList and tempWordList[-1] in filterLetter_num:
            number_count=number_count+1;
            continue;
        if letter in filterLetter_alphabet:# and tempWordList and tempWordList[-1] in filterLetter_alphabet:
            alphabet_count=alphabet_count+1;
            continue;
        if lastdict.get(letter,None):
            tempWordList.append(letter);
            lastdict=lastdict.get(letter);
        else:
            wordSplitedList.append(tempWordList);
            lastdict=worddict.get(letter,None);
            if not lastdict:
                lastdict=worddict;
                wordSplitedList.append([letter])
                tempWordList=list()
            else:
                tempWordList=[letter]
    if tempWordList!=[]:
        wordSplitedList.append(tempWordList);
    return wordSplitedList,(punctuation_count,number_count,alphabet_count);
def findWordInDict(word,judgeWorddict):
    currdict=judgeWorddict;
    for letter in word:
        if not currdict.get(letter):
            return None
        currdict=currdict.get(letter)
    #return currdict.get(valuestr)
    return currdict

def trainWord(word,signStr,judgeWorddict):
    currdict=findWordInDict(word,judgeWorddict);
    if currdict==None:
        return
    if not currdict.get(valuestr,None):
        currdict[valuestr]=dict()
    try:
        currdict[valuestr][signStr]=currdict[valuestr].get(signStr,1)+1;#可扩展
    except KeyError as e:
        pass
def trainSentence(sentence,signStr,worddict):
    for word in splitSentence(sentence,worddict)[0]:
        trainWord(word,signStr,worddict);

def getWordIsSpamProbabilty(word,judgeWorddict):
    currdict=findWordInDict(word,judgeWorddict);
    if not currdict:
        return probabilty_default;
    count_word_in_spam=currdict.get(valuestr,dict()).get(spamStr,1)
    count_word_in_ham=currdict.get(valuestr,dict()).get(hamStr,1)
    probabilty_word_in_spam=count_word_in_spam/(count_word_in_spam+count_word_in_ham)
    probabilty_word_in_ham=count_word_in_ham/(count_word_in_spam+count_word_in_ham)
    return (probabilty_word_in_spam*probabilty_spam)/(probabilty_word_in_spam*probabilty_spam+probabilty_word_in_ham*probabilty_ham)
def getSentenceIsSpamProbabilty(sentence,judgeWorddict):
    result=list()
    tempr=splitSentence(sentence,judgeWorddict)
    for word in tempr[0]:
        result.append((word,getWordIsSpamProbabilty(word,judgeWorddict)))
    return result,tempr[1]


def load_config():
    config=None
    if os.path.isfile(config_file):
        config=json.load(codecs.open(config_file,'r','utf8'))
    return config

def loadJudgeWorddict(filename):
    #filename=config.get("judge_worddict_file")
    return json.load(codecs.open(filename,'r','utf8'))
def saveJudgeWorddict(judgeWorddict,filename):
    #filename=config.get("judge_worddict_file")
    json.dump(judgeWorddict,codecs.open(filename,'w','utf8'))
def trainSample(config):
    judgeWorddict=loadWorddict(config.get('worddict_file'))
    sample_index_list=codecs.open(config.get('sample_index_file'),'r','utf8');
    for item in sample_index_list:
        item=item.split()
        signStr=item[0]
        datapath=item[1]
        datapath=os.path.join('./',datapath)
        if not os.path.isfile(datapath):
            continue;
        with codecs.open(datapath,'r','gbk','ignore') as h:#
            sentence='';
            line=h.read(1024*10)
            while line:
                trainSentence(line,signStr,judgeWorddict)
                line=h.read(1024*10)
    saveJudgeWorddict(judgeWorddict,config.get("judge_worddict_file"))

def getWordDistribute(r):
    # r:[([字],概率)] 例：[(['公','司'],0.91)]
    # return (dict(每种长度的词语出现的次数),dict(概率分别为0-9的词语出现的次数))
    return (Counter(map(lambda x:len(x[0]),r)),Counter(map(lambda x:int(x[1]*10),filter(lambda x:x[0]!=[],r))))
def genLogitProperty(r):
    p=getWordDistribute(r[0])
    word_sum=sum(p[0].values())+sum(p[1].values())+sum(r[1])
    return [p[0].get(1,0)/word_sum,p[1].get(8,0)/word_sum,p[1].get(9,0)/word_sum]
def genLogitTrainSampleFile(config):
    '''根据样本文件生成不同WordIsSpamProbabilty的词的分布，用于逻辑回归拟合'''
    judgeWorddict=loadJudgeWorddict(config.get("judge_worddict_file"))
    sample_index_list=codecs.open(config.get('sample_index_file'),'r','utf8');
    logitTrainSampleFile=codecs.open(config.get("logit_train_sample_file"),'w','utf8')
    logitTrainSampleFile.write('isSpam,w1,p8,p9'+'\n')
    for item in sample_index_list:
        item=item.split()
        signStr=item[0]
        datapath=item[1]
        datapath=os.path.join('./',datapath)
        if spamStr==signStr:
            signStr='1' 
        else:
            signStr='0'
        with codecs.open(datapath,'r','gbk','ignore') as h:
            sentence='';
            line=h.read(1024*10)
            while line:
                #此处代码和trainSample重复，可考虑写个通用函数然后传入lambda实现不同功能
                r=getSentenceIsSpamProbabilty(line,judgeWorddict)
                p=getWordDistribute(r[0])
                word_sum=sum(p[0].values())+sum(p[1].values())+sum(r[1])
                q=genLogitProperty(r)
                #if sum(q)!=0:
                logitTrainSampleFile.write(signStr +','+str(q)[1:-1]+'\n')
                line=h.read(1024*10)
    logitTrainSampleFile.close()
def trainLogitModel(config):
    df=pd.read_csv(config.get("logit_train_sample_file"),sep=',')
    train_cols = df.columns[1:]
    logit=sm.Logit(df['isSpam'],df[train_cols])
    model=logit.fit()
    model.save(config.get("logit_model_file"))
#def saveLogitModel()
def loadLogitModel(config):
    return sm.load(config.get("logit_model_file"))

def train():
    config=load_config()
    print("initWorddict")
    wd=initWorddict(config.get('worddict_txt'))
    print("saveWorddict")
    saveWorddict(wd,config.get('worddict_file'))
    print("trainSample")
    trainSample(config)
    print("genLogitTrainSampleFile")
    genLogitTrainSampleFile(config)
    print("trainLogitModel")
    trainLogitModel(config)
    print("trainLogitModel end")

def test(test_sample_index_file):
    '''根据样本文件生成不同WordIsSpamProbabilty的词的分布，用于逻辑回归拟合'''
    config=load_config()
    m=sm.load("logitModel")
    judgeWorddict=loadJudgeWorddict(config.get("judge_worddict_file"))
    sample_index_list=codecs.open(test_sample_index_file,'r','utf8');
    leftcount=0
    rightcount=0    
    leftritgh=list()
    for item in sample_index_list:
        item=item.split()
        signStr=item[0]
        datapath=item[1]
        datapath=os.path.join('./',datapath)
        if spamStr==signStr:
            signStr=1
        else:
            signStr=0
        with codecs.open(datapath,'r','gbk','ignore') as h:
            sentence='';
            line=h.read(1024*10)
            while line:
                #此处代码和trainSample重复，可考虑写个通用函数然后传入lambda实现不同功能
                r=getSentenceIsSpamProbabilty(line,judgeWorddict)
                p=getWordDistribute(r[0])
                word_sum=sum(p[0].values())+sum(p[1].values())+sum(r[1])
                q=genLogitProperty(r)
                #if sum(q)!=0:
                pre=m.predict(q)
                leftritgh.append((signStr,pre))
                # if (pre[0]<0.51 and signStr==0)or(pre[0]>0.51 and signStr==1):
                #     rightcount=rightcount+1
                # else:
                #     leftcount=leftcount+1
                line=h.read(1024*10)
    return leftritgh
    print(leftcount,rightcount)
if __name__ == '__main__':
    #train()
    test("sample_index1")
    #usage:
    #mconfig=load_config()
    #logitModel=loadLogitModel(mconfig)
    #judgeWorddict=loadJudgeWorddict(mconfig.get("judge_worddict_file"))
    #sentence='吧啦啦啦我在吃饭'
    #r=getSentenceIsSpamProbabilty(sentence,judgeWorddict)
    #logitProperty=genLogitProperty(r)
    #Predict=logitModel.predict(logitProperty)
def tongji(lr,bz):
    c=0
    d=0
    for i in lr:
        if (i[1][0]<bz and i[0]==0)or(i[1][0]>bz and i[0]==1):
            d=d+1
        else:
            c=c+1
    return(c,d)
