import argparse
import concurrent.futures
import cv2
import glob
import os
import shutil
import torch

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


def opt_blocks():
    from basicsr.archs.basicvsr_arch import ConvResidualBlocks
    from basicsr.archs.basicvsrpp_arch import SecondOrderDeformableAlignment
    from basicsr.archs.spynet_arch import BasicModule
    ConvResidualBlocks.forward = torch.compile(ConvResidualBlocks.forward)
    SecondOrderDeformableAlignment.forward = torch.compile(SecondOrderDeformableAlignment.forward)
    BasicModule.forward = torch.compile(BasicModule.forward)


def to_bgr_uint8(outputs):
    x = outputs.squeeze(0).float().clamp_(0, 1).mul_(255.0).round_()
    x = x.permute(0, 2, 3, 1).flip(-1)
    return x.to(torch.uint8).contiguous().to('cpu', non_blocking=True)


def inference(imgs, imgnames, model, save_path, args, pool, pending):
    with torch.no_grad():
        if args.opt_1:
            with torch.autocast('cuda', dtype=torch.float16):
                outputs = model(imgs)
            outputs = outputs.float()
        else:
            outputs = model(imgs)

    # save imgs
    if args.opt_3 and outputs.is_cuda:
        host = to_bgr_uint8(outputs)
        torch.cuda.synchronize()
        frames = [host[i].numpy() for i in range(host.shape[0])]
    else:
        frames = [tensor2img(output) for output in list(outputs.squeeze(0))]

    targets = [os.path.join(save_path, f'{imgname}_BasicVSRPP.png') for imgname in imgnames]
    if pool is not None:
        pending.extend(pool.submit(cv2.imwrite, p, f) for p, f in zip(targets, frames))
    else:
        for p, f in zip(targets, frames):
            cv2.imwrite(p, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/BasicVSRPP_REDS4.pth')
    parser.add_argument(
        '--input_path', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results/BasicVSRPP/000', help='save image path')
    parser.add_argument('--interval', type=int, default=100, help='interval size')
    parser.add_argument('--no-opt-1', dest='opt_1', action='store_false', help='disable optimization 1')
    parser.add_argument('--no-opt-2', dest='opt_2', action='store_false', help='disable optimization 2')
    parser.add_argument('--no-opt-3', dest='opt_3', action='store_false',
                        help='convert output frames on the CPU, as before')
    parser.add_argument('--writers', type=int, default=16, help='PNG writers (0 = serial)')
    parser.add_argument('--no-opt-4', dest='opt_4', action='store_false',
                        help='disable optimization 4')
    parser.set_defaults(opt_1=True, opt_2=True, opt_3=True, opt_4=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.opt_4 and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.opt_1 and device.type != 'cuda':
        args.opt_1 = False
    if args.opt_2 and device.type == 'cuda':
        opt_blocks()

    # set up model
    model = BasicVSRPlusPlus(mid_channels=64, num_blocks=7)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # extract images from video format files
    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSRPP_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSRPP_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')

    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    pool = (concurrent.futures.ThreadPoolExecutor(max_workers=args.writers)
            if args.writers > 0 else None)
    pending = []
    try:
        if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
            imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, args.save_path, args, pool, pending)
        else:
            for idx in range(0, num_imgs, args.interval):
                for fut in pending:
                    fut.result()
                pending.clear()
                interval = min(args.interval, num_imgs - idx)
                imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
                imgs = imgs.unsqueeze(0).to(device)
                inference(imgs, imgnames, model, args.save_path, args, pool, pending)
    finally:
        for fut in pending:
            fut.result()
        if pool is not None:
            pool.shutdown(wait=True)

    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)


if __name__ == '__main__':
    main()
