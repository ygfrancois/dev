Edit by ygfrancois
arcface开源人脸识别库的使用:

1.pull下该库的代码,可基于该代码做检测或者fine-tuning训练:
	https://github.com/deepinsight/insightface
2.配置好运行环境,由于该项目最早是基于python2上的MXNET做的,所以需要配置相同的环境才可以用.
3.到库的根目录下的deploy文件夹下,该文件夹下有训练好的mtcnn-model,和test样例代码.
4.下载作者提供的训练好的模型,笔者下载了算法最先进的arcface的模型:model-r100-arcface-ms1m-refine-v2
5.笔者写了一个real_time_face_recognition,用于做demo,其中调用了作者的mtcnn-model和这个训练好的model-r100-arcface-ms1m-refine-v2.注:arcface模型参数调用是在face_model.py里,real_time_face_recognition里设置的是两组feature的距离小于1就是同一个人,大于1就不是同一个人.


