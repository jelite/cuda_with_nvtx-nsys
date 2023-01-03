import sys
import torch

from models import *

class NsysRange():
    def __init__(self):
        None
    def __enter__(self):
        torch.cuda.cudart().cudaProfilerStart()
    def __exit__(self, type, value, trace_back):
        torch.cuda.cudart().cudaProfilerStop()
class NvtxRange(object):
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        torch.cuda.nvtx.range_push(self.label)
    def __exit__(self, type, value, trace_back):
        torch.cuda.nvtx.range_pop()

def main():
    batch = sys.argv[1]
    model = resnet50()

    # use cuda
    model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    eval_time = get_resnet_time(model, int(batch))
   
    print(f"eval finished : {eval_time}") 
    
    return 0


def get_resnet_time(model, batch):
    model.eval()

    input = torch.rand(batch, 3, 224, 224)
    input = input.cuda()
    with torch.no_grad():
        with NsysRange():
            for step in range(4):
                with NvtxRange(f"iter_#{step}"):
                    output = model(input)
                  
    return 0#elapsed_time


if __name__ == '__main__':
    main()
