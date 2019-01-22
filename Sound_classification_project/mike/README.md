# Mike #
安安各位大大，
肥宅麻煩各位大大 carry 了 嗚嗚 > < 

## Intro of Me ##
- HG lab 315

## Doing ##
### phase 4
- manu_fold_spliter.py 將data  切成 10 fold ，後面所有操作皆以此10 fold 進行。


- self_train.py (利用leo 之前ensemble 出來的 semi data 進行 fine-tune )，並且跟去不同 mdoel  在一開始train 時是否使用 image generator ,mixup 進行不同的 fine-tune (避免要重 acc 重新重 0 開始)，而因為方便起見,參數是直接hard code 寫死在py 檔裡面。


- predict_verified.ipynb 進行  predict  semi  verified data 同時更路徑 進行  predict  testing  data 並最後將每個 fold 的產出交給 leo  進行 stacking 。