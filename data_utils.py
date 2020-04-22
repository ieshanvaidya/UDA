import os
import glob
import shutil
import tarfile
import urllib.request
import io
import pandas as pd
import torch
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset

# Prince Specific
user = os.environ['USER']
CACHE_DIR = f'/scratch/{user}/.cache'
torch.hub.set_dir(CACHE_DIR)
os.environ['TORCH_HOME'] = CACHE_DIR


def prepare_data(task, path='data', sequence_length=512):
    """
    Prepare train, val, test and unsup tsvs
    """
    if not os.path.exists(os.path.join(path, task)):
        os.makedirs(os.path.join(path, task))

    if task == 'IMDb':
        if not os.path.exists(os.path.join(path, task, 'train.tsv')):
            if not os.path.exists(os.path.join(path, task, 'aclImdb')):
                print('Downloading and extracting IMDb data...')
                data_file = os.path.join(path, task, 'aclImdb_v1.tar.gz')
                url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
                urllib.request.urlretrieve(url, data_file)
                with tarfile.open(data_file) as f:
                    f.extractall(path=os.path.join(path, task, '.'))
                os.remove(data_file)

        print('Preparing IMDb data...')
        prepare_imdb_dataset(path)

    if not os.path.exists(os.path.join(path, task, 'unsup_uda.tsv')):
        raise RuntimeError(f'Data prepared. Please run generate_backtranslations.py for {task} task and re-run.')

    print('Generating ids...')

    df_unsup = pd.read_csv(os.path.join(path, task, 'unsup_uda.tsv'), sep='\t')
    df_train = pd.read_csv(os.path.join(path, task, 'train.tsv'), sep='\t')
    df_val = pd.read_csv(os.path.join(path, task, 'val.tsv'), sep='\t')
    df_test = pd.read_csv(os.path.join(path, task, 'test.tsv'), sep='\t')

    texts_a, texts_b = list(df_train['text'].values), None
    labels = list(df_train['label'].values)
    save_ids_data(texts_a, texts_b, labels, os.path.join(path, task, 'train_uda_ids.pt'), sequence_length)

    texts_a, texts_b = list(df_val['text'].values), None
    labels = list(df_val['label'].values)
    save_ids_data(texts_a, texts_b, labels, os.path.join(path, task, 'val_uda_ids.pt'), sequence_length)

    texts_a, texts_b = list(df_test['text'].values), None
    labels = list(df_test['label'].values)
    save_ids_data(texts_a, texts_b, labels, os.path.join(path, task, 'test_uda_ids.pt'), sequence_length)

    texts_a, texts_b = list(df_unsup['text'].values), None
    labels = None
    save_ids_data(texts_a, texts_b, labels, os.path.join(path, task, 'unsup_ori_uda_ids.pt'), sequence_length)

    texts_a, texts_b = list(df_unsup['backtranslation'].values), None
    labels = None
    save_ids_data(texts_a, texts_b, labels, os.path.join(path, task, 'unsup_aug_uda_ids.pt'), sequence_length)

    print('Ids generated and saved.')



def save_ids_data(texts_a, texts_b, labels, path, max_length):
    input_ids, token_type_ids, attention_mask = get_encode(texts_a, texts_b, max_length)
    data_dict = {'input_ids': torch.tensor(input_ids),
                 'token_type_ids': torch.tensor(token_type_ids),
                 'attention_mask': torch.tensor(attention_mask),
                 'labels': torch.tensor(labels) if labels else None}

    torch.save(data_dict, path)



def prepare_imdb_dataset(path='data', train_size=20):
    """
    Prepare IMDb train, val, test, unsupervised tsvs from raw data
    """
    if os.path.exists(os.path.join(path, 'IMDb', 'train.tsv')):
        if os.path.exists(os.path.join(path, 'IMDb', 'val.tsv')):
            if os.path.exists(os.path.join(path, 'IMDb', 'test.tsv')):
                if os.path.exists(os.path.join(path, 'IMDb', 'unsup.tsv')):
                    return

    assert os.path.exists(os.path.join(path, 'IMDb', 'aclImdb'))
    train_data = []
    val_data = []
    test_data = []
    num_pos = train_size // 2
    num_neg = train_size - num_pos
    pos_count = 0
    neg_count = 0

    for label in ['pos', 'neg']:
        for fname in glob.iglob(os.path.join(path, 'IMDb', 'aclImdb', 'train', label, '*.txt')):
            with io.open(fname, 'r', encoding='utf-8') as f:
                line = f.readline()

            lbl = int(label == 'pos')
            if lbl == 1:
                pos_count += 1
                if pos_count <= num_pos:
                    train_data.append([line, lbl])
                else:
                    val_data.append([line, lbl])
            else:
                neg_count += 1
                if neg_count <= num_neg:
                    train_data.append([line, lbl])
                else:
                    val_data.append([line, lbl])


        for fname in glob.iglob(os.path.join(path, 'IMDb', 'aclImdb', 'test', label, '*.txt')):
            with io.open(fname, 'r', encoding='utf-8') as f:
                line = f.readline()

            test_data.append([line, int(label == 'pos')])

    unsup_data = []
    for fname in glob.iglob(os.path.join(path, 'IMDb', 'aclImdb', 'train', 'unsup', '*.txt')):
        with io.open(fname, 'r', encoding='utf-8') as f:
            line = f.readline()
        unsup_data.append(line)

    train_df = pd.DataFrame(train_data, columns=['text', 'label'])
    val_df = pd.DataFrame(val_data, columns=['text', 'label'])
    test_df = pd.DataFrame(test_data, columns=['text', 'label'])
    unsup_df = pd.DataFrame(unsup_data, columns=['text'])

    train_df.to_csv(os.path.join(path, 'IMDb', 'train.tsv'), sep='\t', index=False)
    val_df.to_csv(os.path.join(path, 'IMDb', 'val.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(path, 'IMDb', 'test.tsv'), sep='\t', index=False)
    unsup_df.to_csv(os.path.join(path, 'IMDb', 'unsup.tsv'), sep='\t', index=False)


def load_data_for_backtranslation(task, split='unsup', path='data'):
    """
    Return dataframe and column id for text column for backtranslation
    """
    df = pd.read_csv(os.path.join(path, task, f'{split}.tsv'), sep='\t')
    if task == 'IMDb':
        col_id = 'text'

    else:
        raise NotImplementedError

    return df, col_id



def tokenize_from_left(text_a, text_b, tokenizer, max_length):
    """
    Tokenize text using Bert tokenizer by truncating from left instead of right
    """
    words_a = tokenizer.tokenize(text_a)
    if text_b:
        words_b = tokenizer.tokenize(text_b)
        max_length = max_length - 3 # (CLS - text_a - SEP - text_b - SEP)
    else:
        words_b = []
        max_length = max_length - 2 # (CLS - text_a - SEP)

    while len(words_a + words_b) > max_length:
        if len(words_a) >= len(words_b):
            words_a.pop(0)
        else:
            words_b.pop(0)

    if words_b == []:
        words_b = None

    return words_a, words_b



def get_encode(texts_a, texts_b, max_length):
    """
    Get encoding to feed into Bert Model
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)
    input_ids, token_type_ids, attention_mask = [], [], []

    if texts_b:
        for text_a, text_b in tqdm(zip(texts_a, texts_b), desc='Encode', leave=False):
            words_a, words_b = tokenize_from_left(text_a, text_b, tokenizer, max_length)
            tokens = tokenizer.encode_plus(words_a, words_b, max_length=max_length, pad_to_max_length=True)
            input_ids.append(tokens['input_ids'])
            token_type_ids.append(tokens['token_type_ids'])
            attention_mask.append(tokens['attention_mask'])
    else:
        for text_a in tqdm(texts_a, desc='Encode', leave=False):
            words_a, _ = tokenize_from_left(text_a, None, tokenizer, max_length)
            tokens = tokenizer.encode_plus(words_a, max_length=max_length, pad_to_max_length=True)
            input_ids.append(tokens['input_ids'])
            token_type_ids.append(tokens['token_type_ids'])
            attention_mask.append(tokens['attention_mask'])

    return input_ids, token_type_ids, attention_mask


def get_save_dir(base_dir, name, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        save_dir = os.path.join(base_dir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


class SupervisedDataset(Dataset):
    def __init__(self, data_path):
        super(SupervisedDataset, self).__init__()

        data = torch.load(data_path)
        self.input_ids = data['input_ids']
        self.token_type_ids = data['token_type_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index]


class UnsupervisedDataset(Dataset):
    def __init__(self, ori_data_path, aug_data_path):
        super(UnsupervisedDataset, self).__init__()

        ori_data = torch.load(ori_data_path)
        aug_data = torch.load(aug_data_path)

        self.ori_input_ids = ori_data['input_ids']
        self.ori_token_type_ids = ori_data['token_type_ids']
        self.ori_attention_mask = ori_data['attention_mask']

        self.aug_input_ids = aug_data['input_ids']
        self.aug_token_type_ids = aug_data['token_type_ids']
        self.aug_attention_mask = aug_data['attention_mask']

        assert self.ori_input_ids.shape == self.aug_input_ids.shape

    def __len__(self):
        return self.ori_input_ids.shape[0]

    def __getitem__(self, index):
        return (self.ori_input_ids[index], self.ori_token_type_ids[index], self.ori_attention_mask[index],
                self.aug_input_ids[index], self.aug_token_type_ids[index], self.aug_attention_mask[index])
