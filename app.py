import os
import torch
import torch.nn
import torchvision.transforms as transforms
import gradio as gr
from models import build_model
from config import get_config
import argparse

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer Test script', add_help=False)
    parser.add_argument('--cfg', default='configs/swin/swin_tiny_patch4_window7_224.yaml', type=str, metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', default='output/swin_tiny_patch4_window7_224/default/ckpt_epoch_1.pth',
                        help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    # parser.add_argument("--local_rank", default='0', type=int, help='local rank for DistributedDataParallel')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

# 加载与训练中使用的相同结构的模型
def load_model(checkpoint_path):
    os.environ['LOCAL_RANK'] = '0'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, config = parse_option()
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model.to(DEVICE)
    return model


# 加载图像并执行必要的转换的函数
def process_image(image, image_size):
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    image = trans(image.convert('RGB'))
    return image

# 预测图像类别并返回概率的函数
def predict(image):
    classes = {'0': 'Real', '1': 'Fake'}  # Update or extend this dictionary based on your actual classes
    image = process_image(image, 256)  # Using the image size from training
    with torch.no_grad():
        in_tens = image.unsqueeze(0)
        in_tens = in_tens.cuda()
        probabilities= model(in_tens).softmax(axis=1).squeeze().tolist()
    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}

    return class_probabilities


# 定义到您的模型权重的路径
checkpoint_path = '/mnt/e/swin_out/swin_tiny_patch4_window7_224/default/model_best.pth'
model = load_model(checkpoint_path)
num_classes = 2

# 定义Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Fake vs Real Classifier",
    examples=["examples/fake.png", "examples/real.JPEG"]
)

if __name__ == "__main__":
    iface.launch(share=True, server_name='0.0.0.0', server_port=10086)
