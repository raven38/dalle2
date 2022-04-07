"""
Train a diffusion model on images.
"""

from torch.optim import AdamW
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import argparse

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

from glide_text2im import dist_util, logger
from glide_text2im.image_datasets import load_data_from_file
from glide_text2im.resample import create_named_schedule_sampler
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from glide_text2im.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from glide_text2im.train_util import TrainLoop
from glide_text2im.nn import mean_flat

from glide_text2im.fp16_util import MixedPrecisionTrainer


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Create CLIP model.
    clip_model = create_clip_model(device=dist_util.dev())
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', dist_util.dev()))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', dist_util.dev()))

    logger.log("creating data loader...")
    data = load_data_from_file(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
    prior = TransformerDecoder(decoder_layer, num_layers=6)

    mp_trainer = MixedPrecisionTrainer(
        model=model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    opt = AdamW(
        mp_trainer.master_params, lr=1e-4, weight_decay=0.0
    )

    logger.log("training...")
    for i in range(1000):
        mp_trainer.zero_grad()
        batch, cond = next(data)
        batch = clip_model.image_embeddings(batch)
        cond = clip_model.text_embeddings(cond)
        
        output = prior(batch)
        loss = mean_flat((output - cond)**2)
        mp_trainer.backward(loss)        
        took_step = mp_trainer.optimize(opt)


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
