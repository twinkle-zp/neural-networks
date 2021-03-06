import csv
import re
import time
import pandas as pd
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

#更改新闻发表日期格式，从一月一号算起，当前日期距一月一号的天数；将用户阅读时间改为第几天
def changedate():
    data = csv.reader(open('./data/1data.csv',encoding='ANSI'))
    data_write = csv.writer(open('./data/2data.csv', 'a', newline='', encoding='ANSI'), dialect='excel')
    index = 0
    for line in data:
        if index == 0:
            line.append('新闻发表时间1')
            data_write.writerow(line)  # 存表头
            index = 1
            continue

        # 将浏览时间格式改为是一年中的第几天
        tuptime = time.localtime(int(line[2]))
        readdate = time.strftime("%j",tuptime)
        line[2] = str(readdate)

        #将新闻发表时间改为是一年中的第几天
        pattern = re.compile(r'\d+')    #匹配至少一个数字
        result = pattern.findall(line[5])    #匹配“2014年*月*日”
        month = int(result[1])
        day = int(result[2])
        daynum = 0    #发表时间是第几天
        if(month==1):
            daynum = day
        elif(month == 2):
            daynum = 31 + day
        elif(month == 3):
            daynum = 31 + 28 + day
        elif(month == 4):
            daynum = 31 + 28 + 31 + day
        elif (month == 5):
            daynum = 31 + 28 + 31 + 30 + day
        line.append(str(daynum))
        data_write.writerow(line)

#取每个用户的最后阅读的八条新闻为测试集，其余32条为训练集
def devidesets():
    data = csv.reader(open('./data/2data.csv', encoding='ANSI'))
    train_write = csv.writer(open('./data/3train.csv', 'a', newline='', encoding='ANSI'), dialect='excel')
    test_write = csv.writer(open('./data/4test.csv', 'a', newline='', encoding='ANSI'), dialect='excel')
    user_lenth = 40
    test_lenth = 8   #设定每个用户测试的数量
    count = 0
    index = 0
    for line in data:
        if index == 0:
            train_write.writerow(line)  # 存表头
            test_write.writerow(line)  # 存表头
            index = 1
            continue
        if count < test_lenth:
            test_write.writerow(line)
            count += 1
        elif count < user_lenth:
            train_write.writerow(line)
            count += 1
            if count == user_lenth:
                count = 0



#手动取train和test文件中前100名用户的数据 1+100*32=3201   1+100*8=801



#取出训练集和测试集中所有新闻信息并去重
def extractnews():
    td = csv.reader(open('./data/7data_100user.csv', encoding='ANSI'))
    tdl = list(td)    #将数据转为list
    td2 = csv.reader(open('./data/7data_100user.csv', encoding='ANSI'))
    tdl2 = list(td2)
    delete_list = []
    print(len(tdl))

    #从5train_100user.csv中找重复
    for i in range(1,len(tdl)-1):
        for j in range(i+1,len(tdl)):
            if tdl[i][1]==tdl[j][1]:         #根据新闻id判断是否有重复
                delete_list.append(j)     #此处暂存要删除的下标


    delete_list = list(set(delete_list))   #利用集合去重复
    print(len(delete_list))

    count=0            #删除会改变下标，利用count记录删除的个数
    for i in delete_list:
        del tdl[i-count]
        count += 1

    # 计算每条新闻点击数
    count=0
    for i in tdl:
        for j in tdl2:
            if i[1]==j[1]:
                count += 1
        i.append(str(count))
        count=0

    count=0
    for i in tdl:
        count+=int(i[7])
    print(count)
    print(len(tdl))

extractnews()
















