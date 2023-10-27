import pandas as pd
import subprocess
import os
import sys
import shutil
import json

file_path = "models.xlsx"
xls = pd.read_excel(file_path, sheet_name=None)
sheet_names = xls.keys()

label_files = {
    "Kinetics-400": "/workspace/tools/data/kinetics/label_map_k400.txt",
    "UCF101": "/workspace/tools/data/ucf101/label_map.txt",
    "SthV2": "/workspace/tools/data/sthv2/label_map.txt",
    "Kinetics-700": "/workspace/tools/data/kinetics/label_map_k700.txt",
    "Kinetics-710": "/workspace/tools/data/kinetics710/label_map_k710.txt",
    "SthV1": "/workspace/tools/data/sthv1/label_map.txt",
    "Kinetics-600": "/workspace/tools/data/kinetics/label_map_k600.txt",
    "Moments in Time V1": "/workspace/tools/data/mit/label_map.txt",
    "AVA v2.1": "/workspace/tools/data/ava/label_map.txt",
    "AVA v2.2": "/workspace/tools/data/ava/label_map.txt/",
    "MultiSports": "/workspace/tools/data/multisports/label_map.txt",
    "NTU60-XSub-2D": False,
    "NTU60-XSub-3D": False,
    "FineGYM": "/workspace/tools/data/gym/label_map.txt",
    "NTU60-XSub": False,
    "HMDB51": "/workspace/tools/data/hmdb51/label_map.txt",
    "UCF101Skele": "/workspace/tools/data/ucf101/label_map.txt",
    "Kinetic400": "/workspace/tools/data/kinetics/label_map_k400.txt",
    "NTU120-XSub-2D": False,
    "NTU120-XSub-3D": False,
    "MSRVTT": False,
    "ActivityNet v1.3": "/workspace/tools/data/activitynet/label_map.txt",
}

model_data = {}

for sheet_name in sheet_names:
    label = label_files[sheet_name]
    if not label:
        continue
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    model_list = df["Model"].tolist()
    for model_name in model_list:
        os.mkdir("/workspace/tmp")
        result = subprocess.run(
            [
                "mim",
                "download",
                "mmaction2",
                "--config",
                model_name,
                "--dest",
                "/workspace/tmp",
            ]
        )
        for file in os.listdir("/workspace/tmp"):
            name_list = file.split(".")
            if name_list[-1] == "py":
                os.rename(
                    os.path.join("/workspace/tmp", file),
                    os.path.join("/workspace/models", file),
                )
                config_file = os.path.join("/workspace/models", file)
            elif name_list[-1] == "pth":
                os.rename(
                    os.path.join("/workspace/tmp", file),
                    os.path.join("/workspace/models", file),
                )
                chkpt_file = os.path.join("/workspace/models", file)

        model_data[model_name] = {
            "label_file": label,
            "config_file": config_file,
            "chkpt_file": chkpt_file,
        }

        shutil.rmtree("/workspace/tmp")

        with open("models.json", "w") as fp:
            json.dump(model_data, fp, indent=4)

        print(model_data[model_name])
