from .argparser import parse_arguments


def main():
    arguments = parse_arguments()
    print(arguments)

if __name__ == '__main__':
    main()