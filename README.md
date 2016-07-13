# spamFilter
spamFilter是一个基于贝叶斯算法和逻辑回归的垃圾信息过滤模块。
用法:
    训练模型：
    from spamFilter import train
    train()
    预测:
    from spamFilter import load_config,loadLogitModel,loadJudgeWorddict,getSentenceIsSpamProbabilty,genLogitProperty
    #mconfig=load_config()
    #logitModel=loadLogitModel(mconfig)
    #judgeWorddict=loadJudgeWorddict(mconfig.get("judge_worddict_file"))
    #sentence='吧啦啦啦我在吃饭'
    #r=getSentenceIsSpamProbabilty(sentence,judgeWorddict)
    #logitProperty=genLogitProperty(r)
    #Predict=logitModel.predict(logitProperty)

样本数据下载地址：http://plg.uwaterloo.ca/~gvcormac/treccorpus06/