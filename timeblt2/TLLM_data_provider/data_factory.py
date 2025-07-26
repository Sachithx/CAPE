from torch.utils.data import DataLoader

from TLLM_data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
}


def data_provider(args, config, pretrain=True, flag='train'):
    Data = data_dict[config['data']]
    timeenc = 0 if config['embed'] != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=config['root_path'],
        data_path=config['data_path'],
        flag=flag,
        size=[config['seq_len'], config['label_len'], config['pred_len']],
        features=config['features'],
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=config['seasonal_patterns'] if config['data'] == 'm4' else None,
        pretrain=pretrain
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader