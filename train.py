import os
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
import data_utils
import train_utils
import argparse
from tqdm import tqdm

GLUE_TASKS = ['SST-2']
OTHER_TASKS = ['IMDb']
ALL_TASKS = GLUE_TASKS + OTHER_TASKS

# Prince
user = os.environ['USER']
CACHE_DIR = f'/scratch/{user}/.cache'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=str, help=f'Task | Choose from {", ".join(ALL_TASKS)}')
    parser.add_argument('--name', type=str, default='untitled', help='Experiment name')
    parser.add_argument('--prepare_data', action='store_true', help='Prepare data')
    parser.add_argument('--sequence_length', type=int, default=512, help='Sequence length for Bert')
    parser.add_argument('--total_steps', type=int, default=10000, help='Total number of training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Supervised batch size')
    parser.add_argument('--unsupervised_ratio', type=float, default=3, help='Ratio of unsupervised batch size to supervised')
    parser.add_argument('--uda_coefficient', type=float, default=1, help='Weight for unsupervised loss')
    parser.add_argument('--schedule', type=str, default='linear', help='Schedule for TSA; choose between linear, log, exp')
    parser.add_argument('--uda_softmax_temperature', type=float, default=0.85, help='UDA softmax temperature')
    parser.add_argument('--uda_confidence_threshold', type=float, default=0.45, help='UDA confidence threshold')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--evaluate_every', type=int, default=500, help='Evaluate on the test set every [_] steps')

    args = parser.parse_args()

    assert args.task in ALL_TASKS, 'Invalid task'

    data_path = 'data'
    if args.prepare_data:
        data_utils.prepare_data(args.task, path=data_path, sequence_length=args.sequence_length)

    logs_path = 'logs'
    # Logs path
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    save_path = data_utils.get_save_dir(logs_path, args.name)

    if not torch.cuda.is_available():
        print('GPU not available. Running on CPU...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR).to(device)

    supervised_train_dataset = data_utils.SupervisedDataset(os.path.join(data_path, args.task, 'train_uda_ids.pt'))
    supervised_validation_dataset = data_utils.SupervisedDataset(os.path.join(data_path, args.task, 'val_uda_ids.pt'))
    unsupervised_dataset = data_utils.UnsupervisedDataset(os.path.join(data_path, args.task, 'unsup_ori_uda_ids.pt'), os.path.join(data_path, args.task, 'unsup_aug_uda_ids.pt'))

    supervised_train_dataloader = DataLoader(supervised_train_dataset, batch_size=args.batch_size, shuffle=True)
    supervised_validation_dataloader = DataLoader(supervised_validation_dataset, batch_size=args.batch_size, shuffle=False)
    unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=args.batch_size * args.unsupervised_ratio, shuffle=True)

    supervised_train_dataiter = iter(supervised_train_dataloader)
    unsupervised_dataiter = iter(unsupervised_dataloader)

    supervised_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    unsupervised_criterion = torch.nn.KLDivLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_validation_accuracy = 0

    steps = tqdm(range(1, args.total_steps + 1))
    for step in steps:
        steps.set_description(f'Best validation accuracy: {best_validation_accuracy:.3f}')
        try:
            supervised_batch = next(supervised_train_dataiter)
        except StopIteration:
            supervised_train_dataiter = iter(supervised_train_dataloader)
            supervised_batch = next(supervised_train_dataiter)

        try:
            unsupervised_batch = next(unsupervised_dataiter)
        except StopIteration:
            unsupervised_dataiter = iter(unsupervised_dataloader)
            unsupervised_batch = next(unsupervised_dataiter)

        optimizer.zero_grad()
        total_loss, supervised_loss, unsupervised_loss = train_utils.compute_loss(device, model, supervised_batch,
                                                                                  unsupervised_batch,
                                                                                  supervised_criterion,
                                                                                  unsupervised_criterion, step, args)
        total_loss.backward()
        optimizer.step()

        if not step % args.evaluate_every:
            accuracy = train_utils.evaluate(device, model, supervised_validation_dataloader)
            if accuracy > best_validation_accuracy:
                best_validation_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
