import pandas as pd
import csv
import re

#从原始数据中提取出阅读新闻超过40篇的用户阅读的前40条新闻
def extractdata():
    caixin = csv.reader(open('./data/caixin.csv',encoding='ANSI'))
    data_write = csv.writer(open('./data/1data.csv','a', newline='',encoding='ANSI'),dialect='excel')
    lineList = []
    index = 0
    for line in caixin:
        if index==0:
            data_write.writerow(line)    #存表头
            index = 1
            continue
        if len(lineList)==0:
            if line[3] != '404' and line[4] != 'NULL' and line[5] != 'NULL' and re.search(':', line[5]) != None and re.search('2014年', line[5]) != None:   #取无问题的格式
                line[5] = line[5][0:-5]
                lineList.append(line)
        elif lineList[-1][0]!=line[0]:   #筛选同一个用户阅读的新闻
            if len(lineList)>=40:
                data_write.writerows(lineList[0:40])    #writerows为写入多行,这里取用户阅读的前四十行
            lineList.clear()
            if line[3]!='404' and line[4]!='NULL' and line[5]!='NULL' and re.search(':',line[5])!=None and re.search('2014年', line[5]) != None:
                line[5] = line[5][0:-5]
                lineList.append(line)
        else:
            if line[3] != '404' and line[4] != 'NULL' and line[5] != 'NULL' and re.search(':',line[5])!=None and re.search('2014年', line[5]) != None:
                line[5] = line[5][0:-5]
                lineList.append(line)



