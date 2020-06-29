import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


df = pd.read_csv('./train.dat', sep='\t',header= None)



df.columns = ['id','Tag_score','Topic_score','Author_score','Has_pic','Has_video','Is_huguan','Text_length','Topic_num','label']
#从左到右依次为微博id，微博内容与用户兴趣标签匹配得分，微博内容与用户兴趣话题匹配得分，目标用户与微博作者的相似度得分，
# 微博是否包含图片，微博是否包含视频，目标用户与微博作者是否互相关注，微博文本内容长度，微博中包含的话题链接数，类标签

print(df.info())
print(df.head(10))

plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
Has_pic = df['Has_pic']
Has_pic = [Has_pic.sum(),df.shape[0] - Has_pic.sum()]
print(Has_pic)
explode = [0.01,0.01]
label = ['有图片', '无图片']
plt.pie(Has_pic,explode=explode,labels=label,autopct='%1.1f%%')
plt.title('微博含图片比例')
plt.show()

plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
Has_pic = df['Has_pic']
a = Has_pic.loc[df['label'] == 1]   #推荐
b = Has_pic.loc[df['label'] == 0]   #不推荐

Has_pic = [a.sum(), b.sum(), len(a) - a.sum(), len(b) - b.sum()]
explode = [0.01,0.01,0.01,0.01]
label = ['有图片、用户喜欢', '有图片、用户不喜欢', '无图片、用户喜欢', '无图片、用户不喜欢']
plt.pie(Has_pic,explode=explode,labels=label,autopct='%1.1f%%')
plt.title('微博含图片与用户是否喜欢比例')
plt.show()



plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
Has_video = df['Has_video']
Has_video = [Has_video.sum(),df.shape[0] - Has_video.sum()]
print(Has_video)
explode = [0.01,0.01]
label = ['有视频', '无视频']
plt.pie(Has_video,explode=explode,labels=label,autopct='%1.1f%%')
plt.title('微博含视频比例')
plt.show()

plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
Has_video = df['Has_video']
a = Has_video.loc[df['label'] == 1]   #推荐
b = Has_video.loc[df['label'] == 0]   #不推荐

Has_video = [a.sum(), b.sum(), len(a) - a.sum(), len(b) - b.sum()]
explode = [0.01,0.01,0.01,0.01]
label = ['有视频、用户喜欢', '有视频、用户不喜欢', '无视频、用户喜欢', '无视频、用户不喜欢']
plt.pie(Has_video,explode=explode,labels=label,autopct='%1.1f%%')
plt.title('微博含视频与用户是否喜欢比例')
plt.show()





Is_huguan = df['Is_huguan']

Is_huguan.plot.box(title="互关指数")
plt.grid(linestyle="--", alpha=0.3)
plt.show()

a = Is_huguan.loc[df['label'] == 1]   #推荐
b = Is_huguan.loc[df['label'] == 0]   #不推荐
plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
plt.hist(a, bins = 10, facecolor='green', alpha=0.75)
plt.xlabel('互关指数')
plt.ylabel('频数')
plt.title('用户喜欢的微博中，互关指数')
plt.show()
plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
plt.hist(b, bins = 10, facecolor='blue', alpha=0.75)
plt.xlabel('互关指数')
plt.ylabel('频数')
plt.title('用户不喜欢的微博中，互关指数')
plt.show()


Text_length = df['Text_length']

Text_length.plot.box(title="文本长度")
plt.grid(linestyle="--", alpha=0.3)
plt.show()

a = Text_length.loc[df['label'] == 1]   #推荐
b = Text_length.loc[df['label'] == 0]   #不推荐
plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
plt.hist(a, bins = 10, facecolor='green', alpha=0.75)
plt.xlabel('文本长度')
plt.ylabel('频数')
plt.title('用户喜欢的微博中，文本长度')
plt.xlim((0,200))
plt.show()
plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
plt.hist(b, bins = 10, facecolor='blue', alpha=0.75)
plt.xlabel('文本长度')
plt.ylabel('频数')
plt.title('用户不喜欢的微博中，文本长度')
plt.xlim((0,200))
plt.show()

Topic_num = df['Topic_num']

Topic_num.plot.box(title="链接数")
plt.grid(linestyle="--", alpha=0.3)
plt.show()

a = Topic_num.loc[df['label'] == 1]   #推荐
b = Topic_num.loc[df['label'] == 0]   #不推荐
plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
plt.hist(a, bins = 10, facecolor='green', alpha=0.75)
plt.xlabel('链接数')
plt.ylabel('频数')
plt.title('用户喜欢的微博中，链接数')
plt.xlim((0,10))
plt.show()
plt.figure(figsize = (5,3))
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
plt.hist(b, bins = 10, facecolor='blue', alpha=0.75)
plt.xlabel('链接数')
plt.ylabel('频数')
plt.title('用户不喜欢的微博中，链接数')
plt.xlim((0,10))
plt.show()
