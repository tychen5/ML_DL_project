## Training
### phase1 10-fold training
在phase1，我們各自拿去不同的mfcc feature來做trainging，model的部分分成兩個model，一個為resnet34，另一個為resnet152，在這個階段中我讀進verified data(共3710筆data)並將其split為10-fold，同時將fname與其label給存起來進行training。
在這個階段我們還有做mixup generator(data augmentation)，用以增加training data來訓練model。
model訓練完成後，各自predict semi data(unverified data + testing data)，用以在phase2做co-train。

### phase2 co-train training
在phase2，我們彼此拿predict好的semi data，前提是必須超過threshold(mean+std)，進行在原本的model進行fine-tune

### phase3 self-train
在phase3，我們拿ensemble-verified的semi data，從phase2有改善的model進行fine-tune，並predict testing data
