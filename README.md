# handclassify_tf_for_shengteng

tensorflow train for hand classification 

Totally 9 different gesture types: please read the docx file for details

# Environment 

ubuntu: 16.04

tensorflow: 1.15

python: 3.6, 3.7, 3.8

pip install numpy tqdm opencv-contrib-python


# Dataset

**ClassifyHand-part0**

Each line of data has:

Image_name bbox label

bbox: x, y, w, h

label: orginal label has ID from 0 to 27

# Data Preparation 

/nas/users/wjz/handclassify_data_for_shengteng

**Download dataset to folder: data/shengteng:**

	data/shengteng/ClassifyHand-shengteng-part0
	
	data/shengteng/ClassifyHand-shengteng-part0.txt
        ...

**generate annotation for 9 gestures**

cd to data/shengteng and enter command line:

	python transform_to_9class.py
	
	cat anno-ClassifyHand-shengteng-part0-9class.txt>anno_hand_9class.txt

**Generate training images and annotation**

Go to terminal and enter the command line

	python prepare_data/gen_hand.py --base_num 20 --thread_num 10


# Training 
	
Go to terminal and enter the command line

	python example/train_hand.py --gpus 0 --thread_num 10
	

If your data is gray or infrared image, then add parameter:

	--use_gray True

# Export model

**export pb model**

Go to terminal

Run export script(refer to the code for specific parameter)

	python example/gen_frozen_pb_hand.py --checkpoint models/wjz1_hand/model-300000 --output_graph ./model-wjz1_hand-300000.pb
	
If your data is gray or infrared image, then add parameter:

	--use_gray True

**pb to onnx**

Need to install tf2onnx

	python -m tf2onnx.convert --input model-wjz1_hand-300000.pb --output model-wjz1_hand-300000.onnx \
	       --inputs image:0 --outputs hand_out/flatten/Reshape:0 \
	       --inputs-as-nchw image:0 \
   	       --rename-inputs data --rename-outputs hand \
	       --verbose

# Testing

Go to terminal

Run testing for single image

	python example/test_hand.py


# Others

There are 5 different networks for speed/accuracy tradeoff:

network_wjz1_hand.py

network_wjz2_hand.py

network_wjz3_hand.py

network_wjz4_hand.py

network_wjz5_hand.py
