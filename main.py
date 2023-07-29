from torch.utils.data import DataLoader
import torch
from dataloaders.custom_dataloader import CustomDataset

from utils.utils import get_train_transform
import argparse

from solver import Solver

class TupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(values))


def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')
    parser.add_argument('--annotations_file', type=str, default="annotations.json", help='name of the annotations file')
    parser.add_argument('--dataset_path', type=str, default='./', help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path where to save the trained model')
    parser.add_argument('--results_dir', type=str, default='./', help='path where to save the results')
    parser.add_argument('--writer_path', type=str, default = "./runs/experiments", help= "The path for Tensorboard metrics")
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'evaluate', 'debug'], help = 'net mode (train or test)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help = 'optimizer used for training')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'cross_entropy'], help = 'loss used for training')
    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')
    parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained weights.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help = 'device used')
    parser.add_argument('--manual_seed', type=bool, default=True, help='Use same random seed to get same train/valid/test sets for every training.')
    parser.add_argument('--image_size', type=int, nargs='+', action=TupleAction, help='Image size in the form x1 x2 x3 x4 ...')
    parser.add_argument('--early_stopping', type = int, default = 0, help = "Parameter that controls early stopping. 0 = no early stopping. Values greater than 0 represent the value of patience. Eg: 1 = early stopping with patience 1")
    return parser.parse_args()

def main(args):
    

    BATCH_SIZE = args.batch_size 
    NUM_WORKERS = args.workers

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # use our dataset and defined transformations
    dataset = CustomDataset(
        args, get_train_transform()
    )
    print(len(dataset))

    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    # split the dataset in train and test set
    if args.manual_seed:
        torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:train_len])
    dataset_valid = torch.utils.data.Subset(dataset, indices[train_len : train_len + val_len])
    dataset_test = torch.utils.data.Subset(dataset, indices[train_len + val_len :])

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    data_loader_valid = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    data_loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(len(dataset.indices))
    print(len(dataset_valid.indices))
    print(len(dataset_test.indices))

    print("Device: ", DEVICE)

    # define solver class
    solver = Solver(train_loader=data_loader,
            valid_loader=data_loader_valid,
            test_loader=data_loader_test,
            device=DEVICE,
            args=args)

    # TRAIN model
    if args.mode == "train":
        solver.train()
    elif args.mode == "test":
        solver.test(img_count=50)
    elif args.mode == "evaluate":
        solver.evaluate(0)
    elif args.mode == "debug":
        solver.debug()
    else:
        raise ValueError("Not valid mode")

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)