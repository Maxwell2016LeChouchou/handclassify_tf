# handclassify_tf_for_shengteng

tensorflow训练手势分类---盛腾项目

单帧手势共9种：详细情况请阅读docx文档

# 运行环境

tensorflow: 1.15

其他缺啥装啥

# 数据集


**ClassifyHand-shengteng-part0**

每条记录包含字段如下：

图片名 框 标签

框: x,y,w,h

标签: 原始标签为0-27


# 数据准备

目前数据集放在/nas/users/ZQ/handclassify_data_for_shengteng

**下载数据到data/untouch这个文件夹，解压之后目录如下:**

	data/untouch/ClassifyHand-shengteng-part0
	data/untouch/ClassifyHand-shengteng-part0.txt
        ...

**生成9类的annotation**

控制台进入data/untouch输入命令：

	python transform_to_9class.py
	
	cat anno-ClassifyHand-shengteng-part0-9class.txt>anno_hand_9class.txt

**生成用于训练的图片和annotation**

控制台进入此项目目录

生成emotion数据(具体参数查看代码)

	python prepare_data/gen_hand.py --base_num 20 --thread_num 10


# 训练
	
控制台进入此项目目录

训练emotion模型(具体参数查看代码)

	python example/train_hand.py --gpus 0 --thread_num 10
	
训练灰度图模型需要添加参数

	--use_gray True

# 导出模型

**导出pb模型**

控制台进入此项目目录

运行导出脚本(具体参数查看代码)

	python example/gen_frozen_pb_hand.py --checkpoint models/zq1_hand/model-300000 --output_graph ./model-zq1_hand-300000.pb
	
导出灰度图模型需要添加参数

	--use_gray True

**pb转onnx**

需要安装tf2onnx

	python -m tf2onnx.convert --input model-zq1_hand-300000.pb --output model-zq1_hand-300000.onnx \
	       --inputs image:0 --outputs hand_out/flatten/Reshape:0 \
	       --inputs-as-nchw image:0 \
   	       --rename-inputs data --rename-outputs hand \
	       --verbose

# 测试

控制台进入此项目目录

运行单张图测试示例代码(需要手工修改里面的参数)

	python example/test_hand.py


# 其他说明
