clear all;
close all;
[svmModel, feats, labels] = train_spoof_detector('train_data');



save('svmModel')
