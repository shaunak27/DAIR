import torch
import argparse
import os

def print_results(args):
    for i in range(50):
        file_name = args.load_from + 'epoch_' + str(i) + '.pth'
        if os.path.exists(file_name):
            model = torch.load(file_name)
            print(model[])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read results from .pth files')
    parser.add_argument('--load_from',default = "./models/lambda_10_gamma_0.5/", type = str)
    args = parser.parse_args()
    print_results(args)