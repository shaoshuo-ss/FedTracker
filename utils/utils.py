# -*- coding: UTF-8 -*-
import argparse
import distutils.util


def printf(content, path=None):
    if path is None:
        print(content)
    else:
        with open(path, 'a+') as f:
            print(content, file=f)


def load_args():
    parser = argparse.ArgumentParser()
    # global settings
    parser.add_argument('--start_epochs', type=int, default=0, help='start epochs (only used in save model)')
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=10, help="number of clients: K")
    parser.add_argument('--clients_percent', type=float, default=0.4)
    # parser.add_argument('--num_clients_each_iter', type=int, default=8, help="the fraction of clients: C")
    parser.add_argument('--pre_train', type=lambda x: bool(distutils.util.strtobool(x)), default=False, help="Intiate global model with pre-trained weight.")
    parser.add_argument('--pre_train_path', type=str, default='./result/VGG16/50-20/model_last_epochs_100.pth')
    parser.add_argument('--model_dir', type=str, default='./result/final/VGG16/10-4/')

    # local settings
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--local_optim', type=str, default='sgd', help="local optimizer")
    parser.add_argument('--local_lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--local_momentum', type=float, default=0, help="SGD momentum (default: 0)")
    parser.add_argument('--local_loss', type=str, default="CE", help="Loss Function")
    parser.add_argument('--distribution', type=str, default='iid', help="the distribution used to split the dataset")
    parser.add_argument('--dniid_param', type=float, default=0.8)
    parser.add_argument('--lr_decay', type=float, default=0.999)

    # test set settings
    parser.add_argument('--test_bs', type=int, default=512, help="test batch size")
    parser.add_argument('--test_interval', type=int, default=1)

    # model arguments
    parser.add_argument('--model', type=str, default='ResNet18', help='model name')
    parser.add_argument('--num_classes', type=int, default=200, help="number of classes")

    # other arguments
    parser.add_argument('--dataset', type=str, default='tinyimagenet', help="name of dataset")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--image_size', type=int, default=84, help="length or width of images")
    parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default="./result/test/")
    parser.add_argument('--save', type=lambda x: bool(distutils.util.strtobool(x)), default=True)
    # watermark arguments

    parser.add_argument("--watermark", type=lambda x: bool(distutils.util.strtobool(x)), default=True, help="whether embedding the watermark")
    parser.add_argument("--fingerprint", type=lambda x: bool(distutils.util.strtobool(x)), default=True, help="whether to embed the fingerprints")
    parser.add_argument('--lfp_length', type=int, default=128, help="Bit length of local fingerprints")
    parser.add_argument('--fp_threshold', type=float, default=0.5)
    parser.add_argument('--num_trigger_set', type=int, default=100, help='number of images used as trigger set')
    # parser.add_argument('--embed_layer', type=str, default='model.bn8')
    parser.add_argument('--embed_layer_names', type=str, default='model.bn8')
    parser.add_argument('--freeze_bn', type=lambda x: bool(distutils.util.strtobool(x)), default=True)
    # For CNN4: extractor.norm2 lfp 64
    # For VGG16: model.bn8;model.bn9;model.bn10 lfp 128
    # For ResNet18: layer4.layer1.left.bn1;layer4.layer2.left.bn2 lfp 128
    # For AlexNet: extractor.bn3;extractor.bn4;extractor.bn5 lfp 128 for 10 clients:extractor.bn4;extractor.bn5
    # For ResNet50: conv4.layer1.bottleneck.bn1;conv4.layer1.bottleneck.bn2
    # For ResNet34: layer4.layer1.left.bn1;layer4.layer2.left.bn2

    parser.add_argument('--lambda1', type=float, default=0.0005)
    parser.add_argument('--watermark_max_iters', type=int, default=100)
    parser.add_argument('--fingerprint_max_iters', type=int, default=5)
    parser.add_argument('--lambda2', type=float, default=0.005)
    parser.add_argument('--gem', type=lambda x: bool(distutils.util.strtobool(x)), default=True)

    # other
    parser.add_argument('--attack_type', type=str, default="test")
    parser.add_argument('--attack_client', type=int, default=0)

    args = parser.parse_args()
    args.num_clients_each_iter = int(args.num_clients * args.clients_percent)
    return args
