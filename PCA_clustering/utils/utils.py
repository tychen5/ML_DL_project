import os

def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))
