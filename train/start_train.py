from train.trainer import Trainer
from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser(description="set parameters for train mode")
    parser.add_argument(
        '--model', '-m',
        dest='model',
        default='a2c',
        help='choose the model you want to train',
        metavar="MODEL",
        type=str
    )

    parser.add_argument(
        '--total_timesteps', '-tts',
        dest='total_timesteps',
        default=200000,
        help='set the total_timesteps when you train the model',
        metavar="TOTAL_TIMESTEPS",
        type=int
    )

    return parser

def main():
    parser = create_parser()
    options = parser.parse_args()
    Trainer(models = options.model,
            total_timesteps = options.total_timesteps).train()

if __name__ == "__main__":
    main()