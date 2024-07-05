import argparse

from model import SimCLR2
from data import *

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

PUSH_MEG = False
MSG_TOKEN = ""
COMPILE_MODEL = False


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--data_folder', type=str, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')

    opt = parser.parse_args()

    opt.model_path = './save/SupCon/path_models'
    opt.model_name = '{}_{}_epochs{}_lr_{}'.format("SimCLR", opt.model, opt.epochs, opt.learning_rate)
    
    return opt


def main():
    opt = parse_option()

    seed_everything(114514, workers=True)
    trainer = Trainer(
        accelerator="gpu",
        limit_train_batches=opt.batch_size,
        max_epochs=opt.epochs,
        deterministic=True,
        enable_checkpointing=True,
        default_root_dir="resnet_model"
    )

    SimCLR2_model = SimCLR2()
    SimCLR2_model.set_args(opt)

    trainer.fit(SimCLR2_model, set_loader(opt))

if __name__ == '__main__':
    main()    

# """
# python main_supcon.py --batch_size 2048 
#  --learning_rate 0.5  --temp 0.1 
#  --data_folder EmojiDataset 
#  --save_freq 25
# """




