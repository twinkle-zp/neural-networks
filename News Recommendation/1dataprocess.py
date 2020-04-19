import csv
import re
import time

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

#取每个用户的最后阅读的八条新闻为测试集，其余为训练集
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



devidesets()












