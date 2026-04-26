import os

def main():
    print("=====================================================")
    print("   DukeMTMC-ReID Dataset Download Information       ")
    print("=====================================================")
    print("\nThe `torchreid` package obtained from `pip` does not")
    print("currently support automatic downloading of this dataset")
    print("because the official host links have been taken down.")
    print("\nTo proceed, please perform these manual steps:")
    print("  1. Go to: https://exposing.ai/duke_mtmc/ or find an")
    print("     unofficial mirror (e.g. Kaggle / GitHub).")
    print("  2. Download the 'DukeMTMC-reID.zip' file.")
    print("  3. Extract it directly into this folder:")
    print("     D:\\CCTV(DeepSort+Yolo)\\data\\DukeMTMC-reID\\")
    print("\nExpected structure after extracting:")
    print("  data/")
    print("  └── DukeMTMC-reID/")
    print("      ├── bounding_box_train/")
    print("      ├── bounding_box_test/")
    print("      └── query/")
    print("\nOnce extracted, you can run the training script!")

if __name__ == "__main__":
    main()
