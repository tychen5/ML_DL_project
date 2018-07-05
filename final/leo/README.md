# 大家都去當大神惹QQ
### Sound Classification ###

組長:R06725035 陳廷易，負責書面統籌、規劃、設計流程與模型、統整code、開發方法、上台報告、書面報告

## 套件版本 ##
- scikit-learn 0.19.1
- scipy 1.0.1
- librosa 0.6.0
- tensorflow-gpu 1.8
- keras 2.2
- xgboost 0.71
- lightgbm 2.1

## Final Testing (reproduce) ##
- 因為我們有十種model各model還有十個fold model，在加上stacking model有將近一百五十個model..最大的model甚至超過2GB，model所需空間超過100GB，我們找不到免空可以容納如此大量的地方，因此我們testing改以各model predict全部testing data所得的CSV後，在透過我們phase4及voting的方式進行testing
- 需要的資料有leo/data/map_reverse.pkl、leo/data/sample_submission.csv、leo/data/stacking/lp_model_res/*、leo/data/stacking/nn/*、leo/data/phase4/combine/*、leo/data/phase4/weight_accF.csv、leo/data/stacking/stack_accF.csv

## Training ##
- step0: 利用preprocessing_featureExtracting.ipynb，抽取MFCC feature
- step1: 利用LGD_Phase1_pre_10foldVerified.ipynb，將MFCC feature切成10-fold
(step2,3可略過)
- step4: 利用LGD_Phase1_ResNet.ipynb，進行phase1 training
- step5: 利用LGD_Phase2_CoTrain.ipynb，讀入model並predict全部的semi data (unverified+testing)
- step6: 利用LGD_Phase2_CoTrain.ipynb，製作co-train csv(fname,softmax)，給其他人進行fine-tune
- step7: 利用LGD_Phase2_CoTrain.ipynb其他人model所給的co-train csv，對應回自己model的feature data準備進行fine tune
- step8: 利用LGD_Phase1_ResNet.ipynb進行phase2的co-train
- step9: 利用LGD_Phase3_ens_selfTrain.ipynb進行phase3，進行weighted sum，得出經ensemble verified後的csv (fname,label)
- step10: 利用LGD_Phase1_ResNet.ipynb，利用上一步的data再次進行fine tune
- step11: 利用LGD_Phase4_Stacking.ipynb，訓練Phase4的stage1及stage2的model，並predict出csv進行final testing
