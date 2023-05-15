import torch
from utils.datasets import get_full_dataset
from utils.models import get_model
from utils.test import test_img_top5
from utils.utils import load_args


args = load_args()
# args.attack_type = 'quantization'
model = get_model(args)
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

model.load_state_dict(torch.load('./result/final/ResNet18-tinyimagenet/50-20-84-baseline/model_best.pth'))
train_dataset, test_dataset = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
acc = test_img_top5(model, test_dataset, args)
print(acc)