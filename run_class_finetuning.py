# --------------------------------------------------------
# Original Code from BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets import build_dataset, get_dataset, HyperF
from engine_for_finetuning import train_one_epoch, evaluate, test_pre, test
from utils import NativeScalerWithGradNormCount as NativeScaler, sample_gt, metrics, display_goundtruth
import utils
from scipy import interpolate
import modeling_finetune
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('ConMIM fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--run', type=int, default=1, help="Running times.")
    # Model parameters
    parser.add_argument('--model', default='vit_3_patch', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_name', default='weighted-superpixel_ConMIM', type=str, help='only use for plot')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    # parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    # parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params        #使用mixup会使得线性分类精度降低
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')
    parser.add_argument('--output_dir', default='./output/conmim_finetuned/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    # 高光谱上的参数
    parser.add_argument('--dataset', type=str, default='IndianPines')
    
    parser.add_argument('--folder', type=str, default='../dataset/',
                    help="Folder where to store the "
                         "datasets (defaults to the current working directory).")

    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument('--sampling_mode', type=str, default='random',
                           help="Sampling mode (random sampling or disjoint, default:  fixed)")
    group_dataset.add_argument('--training_percentage', type=float, default=0.1,
                           help="Percentage of samples to use for training")
    group_dataset.add_argument('--train_gt', action='store_true',
                           help="Samples use of training")
    group_dataset.add_argument('--test_gt', action='store_true',
                           help="Samples use of testing")
    group_dataset.add_argument('--validation_percentage', type=float, default=0.1,
                           help="In the training data set, percentage of the labeled data are randomly "
                                "assigned to validation groups.")
    group_dataset.add_argument('--load_data', type=str, default=None,
                           help="Samples use of training")          
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--patch_size', type=int,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
    group_train.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
    parser.add_argument('--depth', default=3, type=int)



    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:     # 分布式框架
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    # print(args)
    RUN = args.run
    DATASET = args.dataset
    FOLDER = args.folder
    LOAD_DATA = args.load_data
    PATCH_SIZE = args.patch_size
    # TRAINING_PERCENTAGE = args.training_percentage
    SAMPLING_MODE = args.sampling_mode
    hyperparams = vars(args)
    img, gt, LABEL_ALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
    N_BANDS = img.shape[-1]
    N_CLASSES = len(LABEL_ALUES)
    args.nb_classes = N_CLASSES
    
    print("opt is : {}".format(args.opt))
    hyperparams.update(
    {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'supervision':'full', 'center_pixel':True})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    print(hyperparams)


    acc_dataset = np.zeros([RUN, 1])
    A = np.zeros([RUN, N_CLASSES-1])
    B = np.zeros([RUN])
    ac = np.zeros([RUN, N_CLASSES-1])
    K = np.zeros([RUN])

    for i in range(RUN):  
        if LOAD_DATA:
            train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
            test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
            train_gt = np.load(train_gt_file, 'r')  
            test_gt = np.load(test_gt_file, 'r')   
        else:     
            train_gt, test_gt = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE, sample_nums=10)

        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(i)
        np.random.seed(i)

        cudnn.benchmark = True
        dataset_train = HyperF(img, train_gt, **hyperparams)

        dataset_test = HyperF(img, test_gt, **hyperparams)
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_test) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_test)

        if global_rank == 0 and args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        if dataset_test is not None:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True
            )
        else:
            data_loader_test = None

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb, 
            init_values=args.layer_scale_init_value,
            n_bands=hyperparams['n_bands'],
            n_classes=hyperparams['n_classes'],
            depth=hyperparams['depth'],
            patch_size=hyperparams['patch_size']
        )

        patch_size = model.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = model.patch_embed.patch_shape

        if args.finetune:
            if args.finetune.startswith('https'): 
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load ckpt from %s" % args.finetune)
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
                print("Expand the shared relative position embedding to each transformer block. ")
                num_layers = model.get_num_layers()
                rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
                for i in range(num_layers):
                    checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

                checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

            all_keys = list(checkpoint_model.keys())
            for key in all_keys:
                if "relative_position_index" in key:
                    checkpoint_model.pop(key)

                if "relative_position_bias_table" in key:
                    rel_pos_bias = checkpoint_model[key]
                    src_num_pos, num_attn_heads = rel_pos_bias.size()
                    dst_num_pos, _ = model.state_dict()[key].size()
                    dst_patch_shape = model.patch_embed.patch_shape
                    if dst_patch_shape[0] != dst_patch_shape[1]:
                        raise NotImplementedError()
                    num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                    src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                    dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                    if src_size != dst_size:
                        print("Position interpolate for %s from %dx%d to %dx%d" % (
                            key, src_size, src_size, dst_size, dst_size))
                        extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                        rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                        def geometric_progression(a, r, n):
                            return a * (1.0 - r ** n) / (1.0 - r)

                        left, right = 1.01, 1.5
                        while right - left > 1e-6:
                            q = (left + right) / 2.0
                            gp = geometric_progression(1, q, src_size // 2)
                            if gp > dst_size // 2:
                                right = q
                            else:
                                left = q

                        dis = []
                        cur = 1
                        for i in range(src_size // 2):
                            dis.append(cur)
                            cur += q ** (i + 1)

                        r_ids = [-_ for _ in reversed(dis)]

                        x = r_ids + [0] + dis
                        y = r_ids + [0] + dis

                        t = dst_size // 2.0
                        dx = np.arange(-t, t + 0.1, 1.0)
                        dy = np.arange(-t, t + 0.1, 1.0)

                        print("Original positions = %s" % str(x))
                        print("Target positions = %s" % str(dx))

                        all_rel_pos_bias = []

                        for i in range(num_attn_heads):
                            z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                            f = interpolate.interp2d(x, y, z, kind='cubic')
                            all_rel_pos_bias.append(
                                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                        rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                        new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                        checkpoint_model[key] = new_rel_pos_bias

            # interpolate position embedding
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = model.pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed

            utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        for name, param in model.named_parameters():
            if name not in ['head.weight', 'head.bias']:
                param.requires_grad = False 
        
        model.to(device)

        model_ema = None
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
            print("Using EMA with decay = %.8f" % args.model_ema_decay)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params:', n_parameters)

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % args.update_freq)
        print("Number of training examples = %d" % len(dataset_train))
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

        num_layers = model_without_ddp.get_num_layers()
        if args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()
        if args.disable_weight_decay_on_rel_pos_bias:
            for i in range(num_layers):
                skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

        if args.enable_deepspeed:
            loss_scaler = None
            optimizer_params = get_parameter_groups(
                model, args.weight_decay, skip_weight_decay_list,
                assigner.get_layer_id if assigner is not None else None,
                assigner.get_scale if assigner is not None else None)
            model, optimizer, _, _ = ds_init(
                args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
            )

            print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
            assert model.gradient_accumulation_steps() == args.update_freq
        else:
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                model_without_ddp = model.module

            optimizer = create_optimizer(
                args, model_without_ddp, skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None,
                get_layer_scale=assigner.get_scale if assigner is not None else None)
            loss_scaler = NativeScaler()

        print("Use step level LR scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print("criterion = %s" % str(criterion))


        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

        if args.eval: 
            test_stats = evaluate(data_loader_test, model, device)
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            )

            if args.output_dir and args.save_ckpt:
                if epoch + 1 == args.epochs:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

            if data_loader_test is not None:
                test_stats = evaluate(data_loader_test, model, device)
                print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
                if max_accuracy < test_stats["acc1"]:
                    max_accuracy = test_stats["acc1"]
                    # 保存best的模型
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

                print(f'Max accuracy: {max_accuracy:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                    log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                    log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        linear_time = time.time() - start_time
        print("linear_time:", linear_time)
        test_stat, prediction, targets, _ = test_pre(data_loader_test, model, device)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stat['acc1']:.1f}%")
        print("prediction和target的长度:",len(prediction),len(targets))       
        results = metrics(prediction, targets, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])
        acc_dataset[i,0] = results['Accuracy']
        print("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
        print("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
        print("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
        print("\nKappa:\n{:.4f}".format(results['Kappa']))
        A[i] = results['F1 scores'][1:]
        print("acc_dataset {}".format(acc_dataset))
        aa = results["aa"][1:] 
        aa_mean = np.nanmean(aa, axis=0)
        ac[i] = aa
        print("ac {}".format(ac))
        B[i] = aa_mean       
        print("aa {}".format(aa))
        K[i] =  results['Kappa']
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write("\nConfusion matrix:\n{}".format(results['Confusion matrix']) + "\n")
            f.write("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
            f.write("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
            f.write("\nKappa:\n{:.4f}".format(results['Kappa']))
            f.write("\naa:\n{}".format(aa))



    OA_std = np.std(acc_dataset)
    OAMean = np.mean(acc_dataset)

    print("OAMean {:.2f} +-{:.2f}".format(OAMean,OA_std))

    aa_std = np.std(B,0)
    aaMean = np.mean(B,0)
    print("aa {:.4f} +-{:.4f}".format(aaMean,aa_std))

    ac_std = np.std(ac,0)
    acMean = np.mean(ac,0)
    kpa_mean = np.mean(K)
    kpa_std = np.std(K)


    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write("OAMean {:.2f} +-{:.2f}".format(OAMean,OA_std) + "\n")
        f.write("aa_run {}".format(B) + "\n")
        f.write("aaMean {:.2f} +-{:.2f}".format(aaMean,aa_std) + "\n") 
        f.write("kappaMean {:.4f} +-{:.4f}".format(kpa_mean,kpa_std) + "\n") 
        f.write("ac:" + "\n")
        for mean, std in zip(acMean, ac_std):     
            f.write("ac {:.2f} ± {:.2f}".format(mean, std) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)