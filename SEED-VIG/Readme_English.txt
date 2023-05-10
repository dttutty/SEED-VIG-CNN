A Multimodal Dataset with EEG and forehead EOG for Vigilance Estimation (SEED-VIG)
This is a subset of SEED (SJTU Emotion EEG Dataset: http://bcmi.sjtu.edu.cn/~seed/) for vigilance estimation called SEED-VIG.

Copyright:
Wei-Long Zheng and Bao-Liang Lu
Center for Brain-like Computing and Machine Intelligence, Department of Computer Science and Engineering, Shanghai Jiao Tong University, China
Key Laboratory of Shanghai Education Commission for Intelligent Interaction and Cognitive Engineering, Shanghai Jiao Tong University, China
Brain Science and Technology Research Center, Shanghai Jiao Tong University, China

If you use SEED-VIG in your research, please cite our following papers:
Wei-Long Zheng and Bao-Liang Lu, A multimodal approach to estimating vigilance using EEG and forehead EOG. Journal of Neural Engineering, 14(2): 026017, 2017.

Xue-Qin Huo, Wei-Long Zheng, and Bao-Liang Lu, Driving Fatigue Detection with Fusion of EEG and Forehead EOG, in Proc. of International Joint Conference on Neural Networks (IJCNN-16), 2016: 897-904.

Nan Zhang, Wei-Long Zheng, Wei Liu, and Bao-Liang Lu,Continuous Vigilance Estimation using LSTM Neural Networks. in Proc. of the 23nd International Conference on Neural Information Processing (ICONIP2016), 2016: 530-537.

If you have any questions about this dataset, please contact Wei-Long Zheng (weilonglive@gmail.com) or Bao-Liang Lu (bllu@sjtu.edu.cn)

Data Desrcitptions:
There are totally 23 experiments. Each experiment contains 885 samples of EEG and EOG. The output is the continuous values from 0 to 1 indicating awake to drowsy states. For evaluation, we separate the entire data from one experiment into five sessions and evaluate the performance with 5-fold cross validation. Correlation coefficient (COR) and root mean square error (RMSE) are used as the final evaluations.

EEG_Feature_2Hz: EEG features (power spectral density: PSD, differential entropy: DE) from the total frequency band (1每50 Hz) with a 2 Hz frequency resolution, psd_movingAveㄛpsd_LDSㄛde_movingAveㄛde_LDS are PSD with moving average, PSD with linear dynamic system, DE with moving average, DE with linear dynamic system. The data format is channel*sample number*frequency bands (17*885*25). The first 1-5 in the first dimension 'channel' are corresponding to temporal brain areas, and the last 7-17 are corresponding to posterior brain areas.

EEG_Feature_5Bands: EEG features (PSD, DE) from five frequency bands: delta (1每4 Hz), theta (4每8 Hz), alpha (8每14 Hz), beta (14每31 Hz), and gamma (31每50 Hz), psd_movingAveㄛpsd_LDSㄛde_movingAveㄛde_LDS are PSD with moving average, PSD with linear dynamic system, DE with moving average, DE with linear dynamic system. The data format is channel*sample number*frequency bands (17*885*5). The first 1-5 in the first dimension 'channel' are corresponding to temporal brain areas, and the last 7-17 are corresponding to posterior brain areas. The 1-5 in the third dimension 'frequency bands' are corresponding to delta (1每4 Hz), theta (4每8 Hz), alpha (8每14 Hz), beta (14每31 Hz), and gamma (31每50 Hz).

EOG_Feature: features_table_icaㄛfeatures_table_minusㄛfeatures_table_icav_minh are forehead EOG features corresponding to different VEO and HEO speration methods using ICA and minus approaches. The data format is sample number*feature dimension (885*36). The order of these 36 EOG features are as follow.

Forehead_EEG: EEG features from forehead electrodes, containing two folders: EEG_Feature_2Hz and EEG_Feature_5Bands. The data format is the same as the above EEG features, but with different channel numbers: 4.

perclos_labels: continuous vigilance labels calculated from eye tracking data.


The order of 36 EOG features:
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