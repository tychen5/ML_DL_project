# sample code
can modify by your need

if load train_Y to pandas:

`Y_train = pd.read_csv('OOXX')`

`Y_train['trans'] = Y_train['label'].map(map_dict)`

`Y_train['onehot'] = Y_train['trans'].apply(lambda x: to_categorical(x,num_classes=41))`

BRs,
Leo
