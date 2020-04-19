import jieba.analyse as analyse

title = "山西司法厅副厅长苏浩去职"
text=   "【财新网】（见习记者王少杰）山西省官场地震余波未止。" \
       "3月25日上午，山西司法厅官网将苏浩名字从“领导介绍”栏目撤下。" \
       "曾任山西公安厅副厅长兼太原公安局局长的苏浩，在2011年11月调至山西省司法厅任排名第三的副厅长。　　" \
       "此前的2014年2月27日，中纪委宣布，山西省人大常委会副主任金道铭因涉嫌严重违纪违法接受调查。" \
       "金道铭曾任中纪委副秘书长兼办公厅主任，2006年8月赴山西，先后任省纪委书记、省委副书记、并曾短期兼政法委书记。" \
       "2014年3月6日，中纪委监察部网站再次发布消息，山西省监察厅副厅长谢克敏涉嫌严重违纪，正接受组织调查。　" \
       "　苏浩现年55岁，山西朔州朔城区人，二级警监。" \
       "1976年2月入伍，1986年10月转业至山西省工商局，1991年4月任山西省朔州市工商局副局长。" \
       "1995年11月，苏浩转入警界，任山西省朔州市公安局副局长，1999年8月后又任朔州市公安局局长、忻州市公安局局长。" \
       "2003年6月，苏浩升任山西省公安厅副厅长，2007年2月至2008年4月兼任大同市公安局局长，2008年4月至2011年兼任太原市公安局局长。" \
       "2011年11月22日，山西省政府常务会议任命苏浩为省司法厅副厅长。　　" \
       "在中央纪委监察部主管的国家风尚网上，2013年1月11日曾发表评论文章《“身边人”有疾，病根在领导》，" \
       "文中称，2011年9月，驾驶“晋O00888”牌照奥迪车的18岁少年苏楠与李双江之子在北京海淀一小区打人，苏楠接受讯问时自称是苏浩之子，" \
       "“苏浩因此陷入舆论漩涡被调离公安系统”。　　" \
       "“打人事件”后，山西省公安厅曾澄清，肇事车主苏楠承认为减轻处理、逃避处罚，编造了是苏浩亲属的理由，" \
       "经认真核查，苏浩与肇事车主没有任何关系。但其后又有媒体报道说，苏浩有两名非婚私生子。　" \
       "　该文称，近几年来，领导干部“家里人”、“身边人”频频出事甚至成为腐败的“中坚力量”早就不是新闻，" \
       "更成为当前违法乱纪、腐败犯罪的一个显著特征，夫人腐败、子女腐败、秘书腐败层出不穷……“综观很多腐败大案，" \
       "我们会发现，‘身边人’出事涉案，常常会拔出萝卜带出泥，因为子女、配偶、司机涉案而牵出领导腐败问题的，那是大概率的事情”。　　苏浩调离山西省公安厅后，续任者李亚力在2012年12月亦被停止山西省公安厅副厅长兼太原市公安局局长职务，接受调查，李亚力被免职与其子李正源涉嫌醉驾殴打执法交警相关。■"
# TF-IDF
title_keyword = analyse.extract_tags(title, topK=2) # topK指定数量，默认20
text_keyword = analyse.extract_tags(title+text, topK=5) # topK指定数量，默认20
print(title_keyword)
print(text_keyword)

