#coding=utf-8

#################################
# 游戏广告识别，主程序模板
# 数据源于腾讯游戏
# file: main_test.py
#################################

import os
import re
from optparse import OptionParser
import pandas as pd
import torch
from transformers import BertTokenizer
from train import SentimentClassifier,getDataloader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###################################
# arg_parser： 读取参数列表
###################################
def arg_parser():
    oparser = OptionParser()

    oparser.add_option("-m", "--model_file", dest="model_file", help="输入模型文件 \
            must be: negative.model", default = None)

    oparser.add_option("-d", "--data_file", dest="data_file", help="输入验证集文件 \
            must be: validation_data.txt", default = None)

    oparser.add_option("-o", "--out_put", dest="out_put_file", help="输出结果文件 \
			must be: result.txt", default = None)

    (options, args) = oparser.parse_args()
    global g_MODEL_FILE
    g_MODEL_FILE = str(options.model_file)

    global g_DATA_FILE
    g_DATA_FILE = str(options.data_file)

    global g_OUT_PUT_FILE
    g_OUT_PUT_FILE = str(options.out_put_file)

###################################
# load_model： 加载模型文件
###################################
def load_model(model_file_name):
	# 创建模型对象
	model = SentimentClassifier()
	# 加载模型文件
	model.load_state_dict(torch.load(model_file_name))
	return model.to(device)

#################################
# 基于数据特征定义两条规则，对最终预测标签做修改
#################################

def rule2(text):
	knowledge = ['滴滴', 'dd', '出售', '要的密', '半价', '低价', '折',
				 '大量出', '出银', '收', '私', '加微信', '加vx', '货到付款']
	flag=False
	for i in knowledge:
		if i in text:
			flag=True
			break
	return  flag

def contact(sentence):
    phone_pattern = r'^1[3-9]\d{9}$'
    qq_pattern = r'[1-9][0-9]{4,10}'
    wechat_pattern = r'[A-Za-z0-9_-]{5,}'
    pattern = f'{phone_pattern}|{qq_pattern}|{wechat_pattern}'
    match = re.search(pattern, sentence)
    return match is not None

###################################
# predict： 根据模型预测结果并输出结果文件，文件内容格式为 label\t text\t
###################################
def predict(model):
	print("predict start.......")
	###################################
	# 预测逻辑和结果输出，( predict_label, content)
	###################################

	tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
	val_loader,texts=getDataloader(g_DATA_FILE,64,tokenizer)
	res=[]
	model.eval()
	with torch.no_grad():

		for (i,batch) in  enumerate(val_loader):

			input_ids = batch['input_ids'].to(device)

			attention_mask = batch['attention_mask'].to(device)

			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			_, predicted = torch.max(outputs, 1)
			res.extend(predicted.tolist())


	df=pd.DataFrame({'label':res,'text':texts})
	for i,t in enumerate(texts):
		if (contact(t) or rule2(t)) and res[i]==0:
			df.loc[i, 'label'] = 1

	df.to_csv(g_OUT_PUT_FILE,sep="\t",index=False)

	print("predict end.......")

	return None

###################################
# main： 主逻辑
###################################
def main():
	print("main start.....")
	
	if g_MODEL_FILE is not None:
		model = load_model(g_MODEL_FILE)
		predict(model)

	print("main end.....")

	return 0

if __name__ == '__main__':
	# print("main start.....")
	arg_parser()
	main()
