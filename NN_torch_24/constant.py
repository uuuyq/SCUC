class CompareConstant:
    # sample
    instance_index = "instance_index"  # 实例索引
    feat = "feat"  # 特征
    scenario = "scenario"  # 场景，没什么用，只是之前考虑将场景作为模型的输入
    cuts_train = "train_cuts"  # 训练时保存的cuts
    x_train = "train_x"  # 训练时保存的x

    # sddip运行至收敛
    time_sddip = "time_sddip"  # 运行时间
    cuts_sddip = "cuts_sddip"  # 与cuts_train 有区别，这里是重新跑sddip至收敛得到的所有cut
    obj_sddip = "obj_sddip"
    LB_sddip = "LB_sddip"
    # read
    time_sddip_read = "time_sddip_read"
    obj_sddip_read = "obj_sddip_read"
    # pred
    time_pred = "time_pred"
    cuts_pred = "cuts_pred"
    x_pred = "x_pred"  # 预测cut需要输入x，这里是计算得到的x
    # re
    time_pred_re = "time_pred_re"
    cuts_pred_re = "cuts_pred_re"
    recalculate_time = "recalculate_time"
    # obj
    obj_nocut = "obj_nocut"
    obj_pred = "obj_pred"
    obj_pred_re = "obj_pred_re"

    # 继续sddip
    time_pred_sddip = "time_pred_sddip"
    LB_pred_sddip = "LB_pred_sddip"
    obj_pred_sddip = "obj_pred_sddip"

    time_pred_re_sddip = "time_pred_re_sddip"
    LB_pred_re_sddip = "LB_pred_re_sddip"
    obj_pred_re_sddip = "obj_pred_re_sddip"

    time_pred_sddip_list = [f"time_pred_sddip_{i}" for i in range(1, 10)]
    obj_pred_sddip_list = [f"obj_pred_sddip_{i}" for i in range(1, 10)]

    time_pred_re_sddip_list = [f"time_pred_re_sddip_{i}" for i in range(1, 10)]
    obj_pred_re_sddip_list = [f"obj_pred_re_sddip_{i}" for i in range(1, 10)]






