警觉度估计前额眼电和脑电数据集
(上海交通大学 计算机科学与工程系 郑伟龙 吕宝粮)
本数据集包含23组实验数据，相关说明和结果发表于“Wei-Long Zheng and Bao-Liang Lu, A multimodal approach to estimating vigilance using EEG and forehead EOG. Journal of Neural Engineering, 14(2): 026017, 2017.”

以下是数据说明：
训练测试细节：每次实验共885样本，依次平均分成5段，时序不打乱，做5折交叉验证，每段177个样本，将五段预测结果拼接与标号分别计算相关系数COR和均方误差RMSE。

EEG_Feature_2Hz：以2Hz带宽计算0-50Hz的脑电特征，psd_movingAve，psd_LDS，de_movingAve，de_LDS分别是PSD滑动平均，PSD线性动力系统平滑，DE滑动平均以及DE线性动力系统平滑特征。数据格式：导联*样本*频带（17*885*25）。其中第一维1-6对应颞叶脑区T区，7-17对应枕部脑区P区。

EEG_Feature_5Bands：以五频段计算脑电特征，psd_movingAve，psd_LDS，de_movingAve，de_LDS分别是PSD滑动平均，PSD线性动力系统平滑，DE滑动平均以及DE线性动力系统平滑特征。数据格式：导联*样本*频带（17*885*5）。其中第一维导联1-6对应颞叶脑区T区，7-17对应枕部脑区P区。第三维频带对应delta (1~4 Hz), theta (4~8 Hz), alpha (8~14 Hz), beta (14~31 Hz), 与 gamma (31~50 Hz)频段的特征。

EOG_Feature： features_table_ica，features_table_minus，features_table_icav_minh分别对应前额眼电不同分离方法。数据格式：样本*特征维度（885*36），36个特征的排列顺序见下。

Forehead_EEG：前额电极提取的脑电信号特征。包含EEG_Feature_2Hz和EEG_Feature_5Bands，数据格式与上面类似，区别是导联数为4.

perclos_labels：利用眼动仪数据计算的PERCLOS标号。

EOG36维特征的排列顺序：
Maximum/mean/sum of blink rate
Maximum/mean/sum of blink amplitude
Power of blink amplitude 
Mean power of blink amplitude 
Maximum/minimum/mean of blink duration

Maximum/mean/sum of saccade rate
Maximum/mean/sum of saccade amplitude
Power of saccade amplitude
Mean power of saccade amplitude
Maximum/minimum/mean of saccade duration

Mean of blink rate variance
Mean of blink amplitude variance
Mean of blink duration variance

Mean of saccade rate variance
Mean of saccade amplitude variance
Mean of saccade duration variance

Blink numbers
Saccade numbers

Maximum of blink rate variance
Maximum of blink amplitude variance
Maximum of blink duration variance
Maximum of saccade rate variance
Maximum of saccade amplitude variance
Maximum of saccade duration variance
