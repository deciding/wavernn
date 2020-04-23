from utils.argutils import print_args
from utils.logger import Logger
from vocoder.train import train
from pathlib import Path
import argparse
import os


#python vocoder_train.py gta_model_libritts_vctk_male dummy --voc_dirs ../datasets/tts_training/training_libritts_wavernn_24k_male ../datasets/tts_training/training_vctk_wavernn_24k_male -m gta_model --mel_dir_name xxx
#python vocoder_train.py gta_model_libritts_vctk_female dummy --voc_dirs ../datasets/tts_training/training_libritts_wavernn_24k_female ../datasets/tts_training/training_vctk_wavernn_24k_female -m gta_model --mel_dir_name xxx

# python vocoder_train.py gta_model dummy --voc_dir ../datasets/tts_training/training_wavernn -m gta_model --mel_dir_name xxx
# gta_model is run_id for saving model state, -m is the saved model path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the vocoder from the synthesizer audios and the GTA synthesized mels, "
                    "or ground truth mels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance. If a model state from the same run ID was previously "
        "saved, the training will restart from there. Pass -f to overwrite saved states and "
        "restart from scratch.")
    parser.add_argument("datasets_root", type=str, help= \
        "Path to the directory containing your SV2TTS directory. Specifying --syn_dir or --voc_dir "
        "will take priority over this argument.")
    parser.add_argument("--syn_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs and the embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/.")
    #parser.add_argument("--voc_dir", type=str, default=argparse.SUPPRESS, help= \
    #    "Path to the vocoder directory that contains the GTA synthesized mel spectrograms. "
    #    "Defaults to <datasets_root>/SV2TTS/vocoder/. Unused if --ground_truth is passed.")
    parser.add_argument("--voc_dirs", nargs='+')
    parser.add_argument("--mel_dir_name", type=str, default="mels", help=\
            "dir name of mel gt inside the training folder of each speaker")
    parser.add_argument("--log_dir", type=str, default="log/", help="tensorboard")
    parser.add_argument("--logger", default=None, help="placeholder")
    parser.add_argument("-m", "--models_dir", type=str, default="vocoder/saved_models/", help=\
        "Path to the directory that will contain the saved model weights, as well as backups "
        "of those weights and wavs generated during training.")
    parser.add_argument("-g", "--ground_truth", action="store_true", help= \
        "Train on ground truth spectrograms (<datasets_root>/SV2TTS/synthesizer/mels).")
    parser.add_argument("-s", "--save_every", type=int, default=1000, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=25000, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("--log_every", type=int, default=1000, help= \
        "Number of steps between log summary.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model and restart from scratch.")
    parser.add_argument("--total_epochs", type=int, default=10000, help= \
        "total epochs to train")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "syn_dir"):
        args.syn_dir = Path(args.datasets_root, "SV2TTS", "synthesizer")
    args.syn_dir = Path(args.syn_dir)
    #if not hasattr(args, "voc_dir"):
    #    args.voc_dir = Path(args.datasets_root, "SV2TTS", "vocoder")
    #args.voc_dir = Path(args.voc_dir)

    # ONLY voc_dirs is useful
    args.voc_dirs = [Path(voc_dir) for voc_dir in args.voc_dirs]
    del args.datasets_root
    args.models_dir = Path(args.models_dir)
    args.models_dir.mkdir(exist_ok=True)

    log_dir=os.path.join(args.log_dir, args.run_id)
    args.logger=Logger(log_dir)

    # Run the training
    print_args(args, parser)
    train(**vars(args))

