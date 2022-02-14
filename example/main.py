from pynewood.config import config


def main():
    p = config()

    p.ugraph = 'insurance.skeleton_fsgnn_th1.0e-05.csv'

    p.log.info("Parameters set")


if __name__ == "__main__":
    main()
