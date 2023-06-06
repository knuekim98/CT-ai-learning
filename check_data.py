# for checking data size

def data_size():
    train_data_size = 0
    val_data_size = 0

    with open('./RIDER Lung CT/train-data/ans.txt', 'r') as f:
        ans = eval(f.read())
        for k in ans.keys():
            train_data_size += len(ans[k])

    with open('./RIDER Lung CT/val-data/ans.txt', 'r') as f:
        ans = eval(f.read())
        for k in ans.keys():
            val_data_size += len(ans[k])
        
    print(f'train_data_size: {train_data_size}')
    print(f'val_data_size: {val_data_size}')

    return train_data_size, val_data_size


if __name__ == '__main__':
    data_size()