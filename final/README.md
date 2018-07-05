# 大家都去當大神惹QQ
### Sound Classification ###

組長:R06725035 陳廷易，負責書面統籌、規劃、設計流程與模型、統整code、開發方法、上台報告、書面報告
組員:R06725041彭証鴻 R06725054蔡鈞 B03902110毛偉倫

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
- 將最終結果輸出至當前目錄下，指令: bash test.sh reproduce.csv

## Training ##
因為training我們這組是四個人分開個別train自己的model，在由我進行統合分配，因此有四種training方式，readme分別於各自資料夾當中

