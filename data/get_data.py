import argparse
import logging
import jetnet
import torch
from pathlib import Path
from tqdm import tqdm


IDX_BKG, IDX_SIG = 0, 1

def main(args):
    logging.info(f"{args=}")
    
    jets_train = {}
    jets_test = {}
    
    path_data = Path(args.path_data)
    path_data.mkdir(parents=True, exist_ok=True)
    
    jet_types = list(set(args.sig_types + args.bkg_types))
    
    for jet_type in tqdm(jet_types):
        # load from jetnet
        jetnet_data = jetnet.datasets.JetNet(
            jet_type=jet_type,
            data_dir=path_data / "hdf5"
        )
        p = torch.from_numpy(jetnet_data.particle_data)
        eta_rel, phi_rel, pt_rel, mask = p.unbind(-1)
        # pT is normalized between -0.5 and 0.5 so the peak pT lies in linear region of tanh
        pt_rel = pt_rel - 0.5
        if args.mask:
            p = torch.stack((eta_rel, phi_rel, pt_rel, mask), dim=-1)
        else:
            p = torch.stack((eta_rel, phi_rel, pt_rel), dim=-1)

        # train-test split
        train_end_idx = int(len(p) * (1 - args.test_size))
        jets_train[jet_type] = p[:train_end_idx]
        jets_test[jet_type] = p[train_end_idx:]
        
    data_train = {
        IDX_BKG: torch.cat([jets_train[jet_type] for jet_type in args.bkg_types], dim=0),
        IDX_SIG: torch.cat([jets_train[jet_type] for jet_type in args.sig_types], dim=0)
    }


    data_test = {
        IDX_BKG: torch.cat([jets_test[jet_type] for jet_type in args.bkg_types], dim=0),
        IDX_SIG: torch.cat([jets_test[jet_type] for jet_type in args.sig_types], dim=0)
    }

    for label, data in zip(('train', 'test'), (data_train, data_test)):
        torch.save(data[IDX_BKG], path_data / f'bkg_{label}.pt')
        torch.save(data[IDX_SIG], path_data / f'sig_{label}.pt')
    
    logging.info(f"Data saved to {path_data}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get data from jetnet')
    parser.add_argument(
        '--path-data', 
        type=str, 
        default='data/particlenet',
        help='Path to data directory.'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Test size.'
    )
    parser.add_argument(
        '--bkg-types', 
        type=str, 
        nargs='+', 
        default=['g', 'q'],
        help='Background jet types.'
    )
    parser.add_argument(
        '--sig-types', 
        type=str, 
        nargs='+', 
        default=['t', 'w', 'z'],
        help='Signal jet types.'
    )
    parser.add_argument(
        '--mask',
        action='store_true',
        default=False, 
        help='Use mask.'
    )
    args = parser.parse_args()
    
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(args)
    
    
    