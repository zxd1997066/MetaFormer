from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
from config import get_inference_config
from models import build_model
from torch.autograd import Variable
from torchvision.transforms import transforms
import numpy as np
import argparse
import time
import os
# from torch._inductor import config
import torch._inductor
# torch._inductor.config.profiler_mark_wrapper_call = True
# torch._inductor.config.cpp.enable_kernel_profile = True

try:
    from apex import amp
except ImportError:
    amp = None

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def model_config(config_path):
    args = Namespace(cfg=config_path)
    config = get_inference_config(args)
    return config


def read_class_names(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    class_list = []

    for l in lines:
        line = l.strip().split()
        # class_list.append(line[0])
        class_list.append(line[1][4:])

    classes = tuple(class_list)
    return classes


class GenerateEmbedding:
    def __init__(self, text_file):
        self.text_file = text_file

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def generate(self):
        text_list = []
        with open(self.text_file, 'r') as f_text:
            for line in f_text:
                line = line.encode(encoding='UTF-8', errors='strict')
                line = line.replace(b'\xef\xbf\xbd\xef\xbf\xbd', b' ')
                line = line.decode('UTF-8', 'strict')
                text_list.append(line)
            # data = f_text.read()
        select_index = np.random.randint(len(text_list))
        inputs = self.tokenizer(text_list[select_index], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=32)
        outputs = self.model(**inputs)
        embedding_mean = outputs[1].mean(dim=0).reshape(1, -1).detach().numpy()
        embedding_full = outputs[1].detach().numpy()
        embedding_words = outputs[0] # outputs[0].detach().numpy()
        return None, None, embedding_words


class Inference:
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.classes = ("cat", "dog")
        self.classes = read_class_names(args.dataset_dir + '/cub-200/attributes.txt')

        self.config = model_config(self.config_path)
        self.model = build_model(self.config)
        self.checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(self.checkpoint, strict=False)
        self.model.eval()
        self.model.to(self.device)

        self.transform_img = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(), # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def infer(self, img_path, meta_data_path):
        # _, _, meta = GenerateEmbedding(meta_data_path).generate()
        # meta = meta.to(self.device)
        # NHWC
        if args.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            print("---- Use NHWC model")
        meta = None
        total_time = 0.0
        total_sample = 0
        if args.compile:
            self.model = torch.compile(self.model, backend=args.backend, options={"freezing": True})
        for i in range(args.num_iter):
            img = Image.open(img_path).convert('RGB')
            img = self.transform_img(img)
            img.unsqueeze_(0)
            img = Variable(img)
            if args.channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
            elapsed = time.time()
            img = img.to(self.device)
            out = self.model(img, meta)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed = time.time() - elapsed

            if args.profile:
                args.p.step()
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_time += elapsed
                total_sample += 1
        _, pred = torch.max(out.data, 1)
        predict = self.classes[pred.data.item()]
        throughput = total_sample / total_time
        latency = total_time / total_sample * 1000
        print('inference latency: %.3f ms' % latency)
        print('inference Throughput: %f images/s' % throughput)
        # print(Fore.MAGENTA + f"The Prediction is: {predict}")
        return predict


def parse_option():
    parser = argparse.ArgumentParser('MetaFG Inference script', add_help=False)
    parser.add_argument('--cfg', type=str, default='./configs/MetaFG_0_224.yaml', metavar="FILE", help='path to config file', )
    # easy config modification
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='datasets and models dir')
    # parser.add_argument('--model-path', default='./datasets/models/metafg_0_1k_224.pth', type=str, help="path to model data")
    # parser.add_argument('--img-path', default='./datasets/cub-200/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0003_8337.jpg',
    #           type=str, help='path to image')
    parser.add_argument('--meta-path', default='', type=str, help='path to meta data')
    # for oob
    # parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--precision', type=str, default='float32', help='precision')
    parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
    # parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_iter', type=int, default=-1, help='num_iter')
    parser.add_argument('--num_warmup', type=int, default=-1, help='num_warmup')
    parser.add_argument('--profile', dest='profile', action='store_true', help='profile')
    parser.add_argument('--quantized_engine', type=str, default=None, help='quantized_engine')
    parser.add_argument('--ipex', dest='ipex', action='store_true', help='ipex')
    parser.add_argument('--jit', dest='jit', action='store_true', help='jit')
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
    parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")
    args = parser.parse_args()
    args.model_path = args.dataset_dir + '/models/metafg_0_1k_224.pth'
    args.img_path = args.dataset_dir + '/cub-200/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0003_8337.jpg'
    return args


if __name__ == '__main__':
    args = parse_option()
    if args.triton_cpu:
        print("run with triton cpu backend")
        import torch._inductor.config
        torch._inductor.config.cpu_backend="triton"
    if args.profile:
        torch._inductor.config.profiler_mark_wrapper_call = True
        torch._inductor.config.cpp.enable_kernel_profile = True
        def trace_handler(p):
            output = p.key_averages().table(sort_by="self_cpu_time_total")
            print(output)
            import pathlib
            timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
            if not os.path.exists(timeline_dir):
                try:
                    os.makedirs(timeline_dir)
                except:
                    pass
            timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                        'MetaFormer-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
            p.export_chrome_trace(timeline_file)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int(args.num_iter/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            args.p = p
            if args.precision == "bfloat16":
                print('---- Enable AMP bfloat16')
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
                    result = Inference(config_path=args.cfg, model_path=args.model_path).infer(
                        img_path=args.img_path, meta_data_path=args.meta_path)
            elif args.precision == "float16":
                print('---- Enable AMP float16')
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
                    result = Inference(config_path=args.cfg, model_path=args.model_path).infer(
                        img_path=args.img_path, meta_data_path=args.meta_path)
            else:
                result = Inference(config_path=args.cfg, model_path=args.model_path).infer(
                        img_path=args.img_path, meta_data_path=args.meta_path)
    else:
        if args.precision == "bfloat16":
            print('---- Enable AMP bfloat16')
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
                result = Inference(config_path=args.cfg, model_path=args.model_path).infer(
                    img_path=args.img_path, meta_data_path=args.meta_path)
        elif args.precision == "float16":
            print('---- Enable AMP float16')
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
                result = Inference(config_path=args.cfg, model_path=args.model_path).infer(
                    img_path=args.img_path, meta_data_path=args.meta_path)
        else:
            result = Inference(config_path=args.cfg, model_path=args.model_path).infer(
                    img_path=args.img_path, meta_data_path=args.meta_path)

    print("Predicted: ", result)

# Usage: python inference.py --cfg 'path/to/cfg' --model_path 'path/to/model' --img-path 'path/to/img' --meta-path 'path/to/meta'
