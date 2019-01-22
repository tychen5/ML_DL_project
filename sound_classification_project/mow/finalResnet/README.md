## Preprocessing

### Split data into n-folds

python3 split_data.py <map.pkl> <X_train path> <num_fold>

python3 get_unverified.py <map.pkl> <X_train path>

## Resnet Training

1) Train mfcc4 resnet18 and resnet152 model using the n-folds data

2) Cotrain mfcc4 resnet18 model with mfcc7 resnet model semi data

3) Self train mfcc4 resnet18 model using 0.8 as threshold semi data

4) Self train mfcc4 resnet18 model using 0.8 as threshold semi data again

5) Cotrain mfcc4 resnet18 model with mfcc6 resnet model semi data

6) Cotrain mfcc4 resnet18 model with mfcc1 resnet18 mixup model semi data

7) Self train mfcc4 resnet18 model using 0.8 as threshold semi data with mixup

