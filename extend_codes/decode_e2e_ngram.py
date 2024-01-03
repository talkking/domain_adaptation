"""Entrypoint of asr model inference, it outputs the kaldi format posterior to stdout"""
import logging
import argparse
from pathlib import Path

import yaml
import torch

from asr.utils.checkpoint import Checkpoint
from asr.data.batch_loader import BatchLoader
from asr.data.dataset import KaldiDataset
from asr.data.collector import SequenceCollector
from asr.utils.common import import_extensions
from asr.utils.split import resolve_data_dir


logger = logging.getLogger(__name__)


def get_parsed_args():
    """Parse arguments for decoding"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir', type=Path, required=True, help='checkpoint-dir')
    parser.add_argument('--ckpt', type=Path, default=None, help='path to checkpoint which you wanna decode with.'
                                                                'If not set, the decoder will use ${dir}/checkpoint')
    parser.add_argument('--testset', type=Path, help='path to the test set location')
    parser.add_argument('--feat-rspec', type=str, help='feat rsepecifier')
    parser.add_argument('--transform', type=str, help='feature transform')
    parser.add_argument('--skip-frame', type=int, help='skip frame')
    # TODO: support inequal left & right splice
    parser.add_argument('--splice', type=int, help='splice value, for both left and right splice')

    parser.add_argument('--stream-size', type=int, default=80, help='Max #sentences per batch')
    parser.add_argument('--frame-limit', type=int, default=4096, help='Max #frames per batch')
    parser.add_argument('--beam', type=int, default=8, help='Search beam')
    parser.add_argument('--penalty', type=float, default=0.6, help='Search beam')
    parser.add_argument('--T', type=float, default=0, help='threshold')
    parser.add_argument('--beta', type=float, default=0.1, help='momentum beta')
    parser.add_argument('--target_nnlm', type=str, help='target domain nnlm')
    parser.add_argument('--target_ngram', type=str, help='target domain ngram')
    args = parser.parse_args()
    return args


def decode(model, data_queue, beam, penalty, target_nnlm, target_ngram, T=0):
    """Run decode

    Read in data batch and writes posterior to stdout.

    Args:
        - model (asr.model.Model): well-trained model
        - data_queue (Iterable[asr.data.Batch]): queue of data batch
    """
    
    model.eval()
    model.load_nnlm2(target_nnlm)
    model.decoder.set_penalty(penalty)
    model.decoder.set_T(T)
    n=0

    one = 0
    total = 0
    with torch.no_grad():
        for batch in data_queue:
            key = batch['uid']
            output = model.decode_e2e(batch, beam)
            ys = output['hyps'].tensor
            xlen = output['hyps'].length
            score = output['scores'][0,0]
            
            for uid, out, length in zip(key, ys, xlen):
                out = out[0, 1:int(length)].cpu().tolist()
                print(uid, *out)
      
            
    
def main():
    """Main entrypoint

    You should call this at your own decode.py after `add_model`
    """
    args = get_parsed_args()

    # Load rspec_template and extension
    config_file = args.dir / 'args.yaml'
    try:
        config = yaml.unsafe_load(config_file.open())
        logger.warning(f'Loadded training config: {config}')
        # TODO: support generic input types
        rspecs_template = config.data['dataset']['data_rspecs']
        resolved_rspecs = resolve_data_dir(rspecs_template, args.testset)
        feat = resolved_rspecs['feat'][0]
        extensions = config.load_extension
    # In case args.yaml not found or load_extension not found
    except (FileNotFoundError, AttributeError):
        logger.warning('Please use newer version of pytorch-asr for better decoding')
        feat = {}
        extensions = ['extend_codes']

    import_extensions(extensions)

    # Build data loader
    arg_feat = {
        'rspec': args.feat_rspec,
        'transform': args.transform,
        'skip_frame': args.skip_frame,
        'splice': (args.splice, args.splice) if args.splice else None,
    }
    # The command line rspecs will override the one extracted from args
    valid_arg_feat = {k: v for k, v in arg_feat.items() if v is not None}
    feat.update(valid_arg_feat)
    data_rspecs = {'feat': [feat]}
    logger.warning(data_rspecs)

    dataset = KaldiDataset(data_rspecs, random_sweep=False, stage='cv')
    collector = SequenceCollector(dataset, minibatch_size=args.stream_size, frame_limit=args.frame_limit, max_length=6000)
    data_queue = BatchLoader(collector)

    if args.ckpt is not None:
        model = Checkpoint.load_model_from_checkpoint(args.ckpt)
    else:
        model = Checkpoint.load_model_from_dir(args.dir)
    if torch.cuda.is_available():
        model.cuda()
    
    # model.cpu()
    # from torch.quantization.qconfig import QConfigDynamic,default_dynamic_quant_observer, default_weight_observer, default_per_channel_weight_observer
    # default_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer, weight=default_weight_observer)
    # default_dynamic_qconfig_channel = QConfigDynamic(activation=default_dynamic_quant_observer, weight=default_per_channel_weight_observer)

    # nn_type = torch.nn.LSTMCell
    # quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear: default_dynamic_qconfig_channel, nn_type: default_dynamic_qconfig}, dtype=torch.qint8)
    
    decode(model, data_queue, args.beam, args.penalty, args.target_nnlm, args.target_ngram, args.T)
    #decode(quantized_model, data_queue, args.beam, args.penalty, args.target_nnlm, args.target_ngram, args.T, args.beta)


if __name__ == '__main__':
    main()
