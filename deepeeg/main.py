#from .argparser import parse_arguments
from dataloader import read_brainvis_from_directory


def main():
    arguments = read_brainvis_from_directory('./')
    train,val = read_brainvis_from_directory('./')
    print(arguments)

if __name__ == '__main__':
    main()