import ast
import os.path
import pickle
from statistics import mean
import torch
from matplotlib import pyplot as plt
from IFR.infer import Infer
from NN_torch_24.config import Config
def fw_calculate(train_data_path, add_cuts_flag=True):
    n_realizations = 6
    num_stage = 24
    n_vars = 13


    result = []
    inference_sddip = Infer(n_stages=num_stage,
                            n_realizations=n_realizations,
                            N_VARS=n_vars,
                            train_data_path=train_data_path,
                            result_path=None
                            )

    for instance_index in range(2, 12):
        if add_cuts_flag:
            # 获取cuts dict
            with open(os.path.join(train_data_path, "cut_processed_pkl", f"{instance_index}_cuts.pkl"), "rb") as f:
                cuts_df = pickle.load(f)
            # cuts_df["value"] = cuts_df["value"].apply(ast.literal_eval)
            cuts = torch.tensor(cuts_df["value"].tolist())
            print("cuts.shape(): ", cuts.shape)
        else:
            cuts = None
        samples = inference_sddip.get_fw_samples(instance_index, fw_n_samples=100)

        obj_list = inference_sddip.forward_obj_calculate(instance_index, samples, cuts)
        result.append(mean(obj_list))

    return result

def main():
    ifr_train_data_path = r"D:\tools\workspace_pycharm\sddip-SCUC-6-24\ifr_data"
    sddip_train_data_path = r"D:\tools\workspace_pycharm\sddip-SCUC-6-24\data_gen_24_bus6_CV\train_data"

    # nocut_obj_list = fw_calculate(sddip_train_data_path, add_cuts_flag=False)
    # sddip_obj_list = fw_calculate(sddip_train_data_path)
    ifr_obj_list = fw_calculate(ifr_train_data_path)



    print(f"ifr_obj_list: {ifr_obj_list}")
    # print(f"sddip_obj_list: {sddip_obj_list}")
    # print(f"nocut_obj_list: {nocut_obj_list}")


    with open(r"ifr_obj_list.pkl", "wb") as f:
        pickle.dump(ifr_obj_list, f)
    # with open(r"sddip_obj_list.pkl", "wb") as f:
    #     pickle.dump(sddip_obj_list, f)
    # with open(r"nocut_obj_list.pkl", "wb") as f:
    #     pickle.dump(nocut_obj_list, f)
    with open(r"sddip_obj_list.pkl", "rb") as f:
        sddip_obj_list = pickle.load(f)
    with open(r"nocut_obj_list.pkl", "rb") as f:
        nocut_obj_list = pickle.load(f)

    x = range(len(ifr_obj_list))  # 横轴：索引或阶段编号

    plt.figure(figsize=(8, 5))
    plt.plot(x, ifr_obj_list, marker='o', label='IFR', linewidth=2)
    plt.plot(x, sddip_obj_list, marker='s', label='SDDiP', linewidth=2)
    plt.plot(x, nocut_obj_list, marker='^', label='SDDiP (no cuts)', linewidth=2)

    plt.xlabel("Iteration / Time step")
    plt.ylabel("Objective Value")
    plt.title("Comparison of IFR and SDDiP Objective Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
