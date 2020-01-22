from argparse import ArgumentParser
from src.dsb import train, test

parser = ArgumentParser(description='Deep End-to-End Calibration')
parser.add_argument("--mode", type=str, default="train", help="operation to perform on model")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs run (default: 200)")
parser.add_argument("--batch_sizes", type=tuple, default=(4, 4),
                    help="batch size for (training, validation)")
parser.add_argument("--data_dir", type=str, help="location of data", default="data")
parser.add_argument("--log_dir", type=str, help="location of logging", default="log")
parser.add_argument("--fractions", type=tuple, help="how much of the data to use for (training, validation)",
                    default=(0.9, 0.1))
parser.add_argument("--workers", type=int, help="number of data loading workers", default=2)
parser.add_argument("--use_gpu", type=bool, help="whether to parallelise(default: True)", default=True)
args = parser.parse_args()

if args.mode == "train":
    train(n_epochs=args.n_epochs, batch_sizes=args.batch_sizes, data_dir=args.data_dir,
          log_dir=args.log_dir, fractions=args.fractions, workers=args.workers, use_gpu=args.use_gpu)


elif args.mode == "test":
    test(batch_size=args.batch_sizes[-1], data_dir=args.data_dir, log_dir=args.log_dir,
         workers=args.workers, use_gpu=args.use_gpu)
