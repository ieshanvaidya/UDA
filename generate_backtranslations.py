import os
import torch
import nltk
import data_utils
import argparse
import pickle
import shutil

GLUE_TASKS = ['SST-2']
OTHER_TASKS = ['IMDb']
ALL_TASKS = GLUE_TASKS + OTHER_TASKS


def split_sent_by_punc(sent, punc, offset):
    """
    Further split sentences when nltk's sent_tokenizer fail.
    """
    sent_list = []
    start = 0
    while start < len(sent):
        if punc:
            pos = sent.find(punc, start + offset)
        else:
            pos = start + offset
        if pos != -1:
            sent_list += [sent[start: pos + 1]]
            start = pos + 1
        else:
            sent_list += [sent[start:]]
            break
    return sent_list



def split_paragraph(paragraph, sent_tokenizer):
    """
    Split a paragraph into sentences. Adapted as per:
    https://github.com/google-research/uda/blob/master/back_translate/split_paragraphs.py

    Parameters
    ----------
    paragraph: str
        Paragraph to be split into sentences
    sent_tokenizer: nltk.tokenize.sent_tokenizer
        Tokenizer
    """

    text = paragraph.strip()
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    sent_list = sent_tokenizer(text)
    has_long = False
    for split_punc in [".", ";", ",", " ", ""]:
        if split_punc == " " or not split_punc:
            offset = 100
        else:
            offset = 5
        has_long = False
        new_sent_list = []
        for sent in sent_list:
            if len(sent) < 300:
                new_sent_list += [sent]
            else:
                has_long = True
                sent_split = split_sent_by_punc(sent, split_punc, offset)
                new_sent_list += sent_split
        sent_list = new_sent_list
        if not has_long:
            break

    return new_sent_list



def preprocess_data(texts):
    """
    Preprocesses data by splitting paragraphs as per:
    https://github.com/google-research/uda/blob/master/back_translate/split_paragraphs.py

    Parameters
    ----------
    texts: list
        List of paragraphs

    Returns
    -------
    sentences: list
        List of sentences
    indices: list
        List of indices indicating start of paragraphs in sentences
    """
    sentences = []
    indices = []
    tokenizer = nltk.tokenize.sent_tokenize
    count = 0

    for text in texts:
        indices.append(count)
        splits = split_paragraph(text, tokenizer)
        sentences += splits
        count += len(splits)

    return sentences, indices



def combine_sentences(sentences, indices):
    """
    Rejoin sentences based on indices. Refer to outout of preprocess_data().

    Parameters
    ----------
    sentences: list
        List of sentences, parts of which need to combined as paragraphs
    indices: list
        List of indices indicating start of paragraphs in sentences

    Returns
    -------
    combined: list
        List of paragraphs
    """
    combined = []
    for i in range(1, len(indices)):
        datum = " ".join([x.strip() for x in sentences[indices[i - 1]:indices[i]]])
        combined.append(datum)

    combined.append(" ".join([x.strip() for x in sentences[indices[-1]:]]))
    return combined


if __name__ == '__main__':
    # Prince Cluster Specific | To avoid downloads in Home
    user = os.environ['USER']
    CACHE_DIR = f'/scratch/{user}/.cache'
    torch.hub.set_dir(CACHE_DIR)
    os.environ['TORCH_HOME'] = CACHE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=str, help=f'Task | Choose from {", ".join(ALL_TASKS)}')
    parser.add_argument('--beam', type=int, default=1, help='Beam size for translation')

    args = parser.parse_args()

    assert args.task in ALL_TASKS, 'Invalid task'

    data_path = 'data'

    assert os.path.exists(data_path), f'{data_path} folder not found'

    df, col_id = data_utils.load_data_for_backtranslation(task=args.task, split='unsup', path=data_path)

    print('Pre-processing data...')
    splits, indices = preprocess_data(df[col_id].values)

    if not torch.cuda.is_available():
        print('GPU not available. Running on CPU...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load an En-De Transformer model trained on WMT'19 data:
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe',
                           formatce_reload=True)
    en2de = en2de.to(device)
    print('Generating forward translations...')
    translations = en2de.translate(splits, beam=args.beam)

    del en2de
    torch.cuda.empty_cache()

    # Load an De-En Transformer model trained on WMT'19 data:
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe',
                           formatce_reload=True)
    de2en = de2en.to(device)

    print('Generating backward translations...')
    back_translations = de2en.translate(translations, beam=args.beam)

    print('Combining back-translations and saving...')
    combined = combine_sentences(back_translations, indices)
    df['backtranslation'] = combined
    df.to_csv(os.path.join(data_path, args.task, f'unsup_uda.tsv'), sep='\t', index=False)

    # Copy train and test for name consistency
    # shutil.copy(os.path.join(data_path, args.task, 'train.tsv'), os.path.join(data_path, args.task, 'train_uda.tsv'))
    # shutil.copy(os.path.join(data_path, args.task, 'test.tsv'), os.path.join(data_path, args.task, 'test_uda.tsv'))

    print(f'Done. Data saved at {os.path.join(data_path, args.task, "unsup_uda.tsv")}')
