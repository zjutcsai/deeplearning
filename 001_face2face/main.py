# -*- coding: utf-8 -*-
import argparse
from tools.train_script import train_net
from tools.test_script import test_net
    
def parse_argments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-step', dest='max_step', type=int, default=80000, help='Max train iterations')
    parser.add_argument('--save-step', dest='save_step', type=int, default=1000, help='Model save iterations')
    parser.add_argument('--vis-step', dest='vis_step', type=int, default=200, help='Validate iteration in training')
    parser.add_argument('--log-step', dest='log_step', type=int, default=50, help='Logging iteration in training')
    parser.add_argument('--vis-dir', dest='vis_dir', type=str, default='output', help='Visualize directory')
    parser.add_argument('--model-dir', dest='model_dir', type=str, default='model', help='Model save dirctory')
    parser.add_argument('--train-dir', dest='train_dir', type=str, default='data/origin/train', help='Training data directory')
    parser.add_argument('--test-dir', dest='test_dir', type=str, default='data/origin/val', help='Validating data directory')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('--image-size', dest='image_size', type=int, default=256)
    parser.add_argument('--net-name', dest='net_name', type=str, default='Pix2Pix')
    parser.add_argument('--mode', dest='mode', choices=['train', 'test'], default='train')
    parser.add_argument('--test-video', dest='test_video')
    args = parser.parse_args()
    return args

def train(args):
    train_net(args)
    
def test(args):
    test_net(args)
    

if __name__ == '__main__':
    args = parse_argments()
    if args.mode == 'train':
        train(args)
    else:
        test(args)