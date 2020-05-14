

def main():
    sbd_threshold = args.sbd_threshold
    params = TransNetParams()
    transnet_cp_path = cur_folder / "TransNet/model/transnet_model-F16_L3_S2_D256"
    params.CHECKPOINT_PATH = str(transnet_cp_path.absolute())
    net = TransNet(params)

if __name__ == '__main__':
    main()