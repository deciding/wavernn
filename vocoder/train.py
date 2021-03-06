from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.display import stream, simple_table
#from vocoder.gen_wavernn import gen_testset
from torch.utils.data import DataLoader
#from pathlib import Path
#from typing import List
from torch import optim
import torch.nn.functional as F
import vocoder.hparams as hp
import numpy as np
import time

#CUDA_VISIBLE_DEVICES=3 python vocoder_train.py gta_model dummy --voc_dir ../datasets/tts_training/training_wavernn/ -m gta_model

def train(run_id='',
        syn_dir=None, voc_dirs=[], mel_dir_name='', models_dir=None, log_dir='',
        ground_truth=False,
        save_every=1000, backup_every=1000, log_every=1000,
        force_restart=False, total_epochs=10000, logger=None):
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    # Instantiate the model
    print("Initializing the model...")
    model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims, # 512
        fc_dims=hp.voc_fc_dims, # 512
        bits=hp.bits, # 9
        pad=hp.voc_pad, # 2
        upsample_factors=hp.voc_upsample_factors, # (3, 4, 5, 5) -> 300, (5,5,12)?
        feat_dims=hp.num_mels, # 80
        compute_dims=hp.voc_compute_dims, # 128
        res_out_dims=hp.voc_res_out_dims, # 128
        res_blocks=hp.voc_res_blocks, # 10
        hop_length=hp.hop_length, # 300
        sample_rate=hp.sample_rate, # 24000
        mode=hp.voc_mode # RAW (or MOL)
    ).cuda()

    # hp.apply_preemphasis in VocoderDataset
    # hp.mu_law in VocoderDataset
    # hp.voc_seq_len in VocoderDataset
    # hp.voc_lr in optimizer
    # hp.voc_batch_size for train

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups:
        p["lr"] = hp.voc_lr # 0.0001
    loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss

    # Load the weights
    model_dir = models_dir.joinpath(run_id) # gta_model/gtaxxxx
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir.joinpath(run_id + ".pt") # gta_model/gtaxxx/gtaxxx.pt
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of WaveRNN from scratch\n")
        model.save(str(weights_fpath), optimizer)
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(str(weights_fpath), optimizer)
        print("WaveRNN weights loaded from step %d" % model.step)

    # Initialize the dataset
    #metadata_fpath = syn_dir.joinpath("train.txt") if ground_truth else \
    #    voc_dir.joinpath("synthesized.txt")
    #mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    #wav_dir = syn_dir.joinpath("audio")
    #dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)
    #dataset = VocoderDataset(str(voc_dir), 'mels-gta-1099579078086', 'audio')
    dataset = VocoderDataset([str(voc_dir) for voc_dir in voc_dirs], mel_dir_name, 'audio')
    #test_loader = DataLoader(dataset,
    #                         batch_size=1,
    #                         shuffle=True,
    #                         pin_memory=True)

    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])

    for epoch in range(1, total_epochs):
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_vocoder,
                                 batch_size=hp.voc_batch_size,
                                 num_workers=30,
                                 shuffle=True,
                                 pin_memory=True)
        start = time.time()
        running_loss = 0.

        # start from 1
        for i, (x, y, m) in enumerate(data_loader, 1):
            # cur [B, L], future [B, L] bit label, mels [B, D, T]
            x, m, y = x.cuda(), m.cuda(), y.cuda()

            # Forward pass
            # [B, L], [B, D, T] -> [B, L, C]
            y_hat = model(x, m)
            if model.mode == 'RAW':
                # [B, L, C] -> [B, C, L, 1]
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y = y.float()
            # [B, L, 1]
            y = y.unsqueeze(-1)

            # Backward pass
            # [B, C, L, 1], [B, L, 1]
            # cross_entropy for RAW
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            if backup_every != 0 and step % backup_every == 0 :
                model.checkpoint(str(model_dir), optimizer)

            if save_every != 0 and step % save_every == 0 :
                model.save(str(weights_fpath), optimizer)

            if log_every != 0 and step % log_every == 0 :
                logger.scalar_summary("loss", loss.item(), step)

            total_data=len(data_loader)

            speed=speed
            avg_loss=avg_loss
            k=k
            total_data=total_data
            msg = ("| Epoch: {epoch} ({i}/{total_data}) | " +\
                "Loss: {avg_loss:.4f} | {speed:.1f} " +\
                "steps/s | Step: {k}k | ").format(**vars())
            stream(msg)


        #gen_testset(model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
        #            hp.voc_target, hp.voc_overlap, model_dir)
        print("")
