# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


def main():
    sliding_window = 16
    for l in range(1, 44):
        print("Length: ", l)
        if l % sliding_window == 0:
            print("\tlinear")
        elif (l > sliding_window):
            ll = l % sliding_window
            print(f"\tphase 1: {ll} -> {ll + sliding_window - ll}")
            print(f"\tphase 2: 0 ->{ll}")
        else:
            print(f"\tphase 1: {l}")

if __name__ == "__main__":
    main()