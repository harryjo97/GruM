import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer3D
from sampler import Sampler_3D_mol

def main(work_type_args):
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer3D(config)
        trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()))


    # -------- Generation --------
    elif work_type_args.type == 'sample':
        sampler = Sampler_3D_mol(config)
        sampler.sample()
        
    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
