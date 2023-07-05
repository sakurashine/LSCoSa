from ops.argparser import  argparser
from ops.Config_Environment import Config_Environment
def main(args):
    #config environment
    ngpus_per_node=Config_Environment(args)
    from training.main_worker import main_worker
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)  # 0,1
if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    # print(args)
    main(args)
