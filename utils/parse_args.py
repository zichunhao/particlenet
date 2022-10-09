import argparse
import torch

def parse_args() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--log-file", type=str, default="", help='log file name - default is name of file in outs/ ; "stdout" prints to console')
    parser.add_argument("--log", type=str, default="INFO", help="log level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument("--path-results", type=str, required=True, help="path to results directory")

    # data settings
    parser.add_argument('--path-data-train-sig', type=str, required=True, help='path to training signal data')
    parser.add_argument('--path-data-train-bkg', type=str, required=True, help='path to training background data')
    parser.add_argument('--path-data-test-sig', type=str, required=True, help='path to testing signal data')
    parser.add_argument('--path-data-test-bkg', type=str, required=True, help='path to testing background data')
    parser.add_argument("--num-particles", type=int, default=30, help="num of particles in each jet")
    
    # model settings
    parser.add_argument("--load-model", default=False, action="store_true", help="load a pretrained model")
    parser.add_argument("--path-model-weights", type=str, default="", help="path to model weights, used if --load-model is called")
    
    # training settings
    parser.add_argument("--start-epoch", type=int, default=0, help="which epoch to start training on (only makes sense if loading a model)")
    parser.add_argument('--mask', default=False, action='store_true', help="use masking")
    parser.add_argument("--num-epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=64, help="batch size for testing")
    parser.add_argument("--optimizer", type=str, default="adamw", help="pick optimizer", choices=['adam', 'rmsprop', 'adamw'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--scheduler', default=False, action='store_true', help="use one cycle LR scheduler")
    parser.add_argument('--lr-decay', type=float, default=0.1)
    parser.add_argument('--cycle-up-num-epochs', type=int, default=8)
    parser.add_argument('--cycle-cooldown-num-epochs', type=int, default=4)
    parser.add_argument('--cycle-max-lr', type=float, default=3e-3)
    parser.add_argument('--cycle-final-lr', type=float, default=5e-7)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    args = parser.parse_args()

    args.node_feat_size = 4 if args.mask else 3  # 3 for (eta, phi, pT) and 4 for (eta, phi, pT, mask)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dtype = torch.float64

    return args