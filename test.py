import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
 
from enum import Enum
from train import * 

def parseParam():
    class LossTYPE(Enum):
        focal = 'focal'
        KLD = 'KLD'
        def __str__(self):
            return self.value
    parser = argparse.ArgumentParser(description='Build model')
    parser.add_argument('-heterodir', type=str, required=True, help='Please input the heterodir.')
    parser.add_argument('-writerdir', type=str, required=True, help='Please input the writedir.')
    parser.add_argument('-losstype', type=LossTYPE, choices=list(LossTYPE), required=True, help='Please input the loss type.')
    parser.add_argument('-mask', nargs='+', default=[], type=int,  help='mask features')
    parser.add_argument('-name', type=str, default="Anonymous_",  help="please input the name of task")
    parser.add_argument('-device', type=int, default=0,  help="Please input the gpu id of using device.")
    parser.add_argument('-n_hid', type=int, default=400, required=True, help="Please input the dimension of features.")
    parser.add_argument('-n_layers', type=int, default=5, required=True, help="Please input the dimension of features.")
    parser.add_argument('-load_Path', type=str, default=None, required=True, help="Please input the path of pre-trained model.")
    args = parser.parse_args()
    return args



if __name__=='__main__':
    log = Logger(sys.path[-1])
    args = parseParam()
    log.info('using ' + args.heterodir+' as hetero dir.')
    log.info('using ' + args.writerdir+' as writer dir.')
    log.info('using ' + str(args.losstype)+' as losstype.')
    log.info('using ' + str(args.name)+' as name.')
    log.info('using ' + str(args.device) + ' as gpu id.')
    log.info('using ' + str(args.n_hid) + ' as hidder feature dim .')
    log.info('using ' + str(args.n_layers) + ' as the layers.')
    log.info('using ' + str(args.load_Path) + ' as pretrain model.')
    
    transformer = MeshTransformer(
                                            args,
                                            args.heterodir,
                                            args.writerdir,
                                            loss_type=str(args.losstype),
                                            lr_schedual_type='Step',
                                            task_name=args.name,
                                            device=args.device,
                                            n_hid=args.n_hid,
                                            n_layers=args.n_layers, 
                                            load_Path=args.load_Path
                                        )
    for i in range(10):
        transformer.test(0, folder=i)