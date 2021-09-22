import os
import time
import argparse
from mpi4py import MPI


parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--arch', dest='arch', default='SiamDW',
                    help='architecture of model')
parser.add_argument('--start_epoch', default=30, type=int, required=True, help='test end epoch')
parser.add_argument('--end_epoch', default=50, type=int, required=True,
                    help='test end epoch')
parser.add_argument('--gpu_nums', default=4, type=int, required=True, help='test start epoch')
parser.add_argument('--anchor_nums', default=5, type=int, help='anchor numbers')
parser.add_argument('--threads', default=16, type=int, required=True)
parser.add_argument('--dataset', default='VOT0219', type=str, help='benchmark to test')
parser.add_argument('--align', default='False', type=str, help='align')
args = parser.parse_args()

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % args.gpu_nums
node_name = MPI.Get_processor_name()  # get the name of the node
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)

# run test scripts -- two epoch for each thread
for i in range(2):
    arch = args.arch
    dataset = args.dataset
    try:
        epoch_ID += args.threads   # for 16 queue
    except:
        epoch_ID = rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch

    if epoch_ID > args.end_epoch:
        continue

    resume = 'snapshot/checkpoint_e{}.pth'.format(epoch_ID)
    print('==> test {}th epoch'.format(epoch_ID))

    if arch == 'Ocean':
        os.system('python ./tracking/test_ocean.py --arch {0} --resume {1} --dataset {2} --align {3} --epoch_test True'.format(arch, resume, dataset, args.align))
    elif arch == 'SiamDW':
        os.system('python ./tracking/test_siamdw.py --arch {0} --resume {1} --dataset {2} --epoch_test True'.format(arch, resume, dataset))
