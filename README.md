# CSM-chatchat
基于langchain-chatchat 0.3的项目进行修改，通过python的方式进行部署

# 该项目原始地址 https://github.com/chatchat-space/Langchain-Chatchat

## 痛点：原项目通过poetry发布项目，过程复杂，不利于修改代码，现修改为python工程，可以通过python启动，同时去掉一些不需要的tool等功能



## 1. 环境安装说明
conda create -n CSM-chatchat python=3.9
conda activate CSM-chatchat
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

## 2. 项目启动说明
- python cli.py init # 生成配置文件yaml
- python cli.py kb -r # 根据data中的samples中的文件，构建向量进行插入
- python cli.py start - a # 启动api服务和启动web服务
