import os
from os.path import exists

import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.basic import color

import utils
import numpy as np
import time
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
# from models.blip_retrieval_tta import get_all_text_embeds, get_all_image_embeds
import models.blip_tta as BLIP
import models.clip_tta as CLIP
import models.clip_reid_tta as CLIP_Reid
import utils
from losses import *
from ddp import *
from confidence_visual import *
from plot_noise import linestyles


@torch.no_grad()
def evaluation(model, test_loader_text, test_loader_image, device, args, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = test_loader_text.dataset.text   
    # text_embeds=get_all_text_embeds(test_loader_text, model, device, args)
    # image_embeds=get_all_image_embeds(test_loader_image, model, device, args)
    if config['backbone']=='blip':
        text_embeds=BLIP.get_text_embeds_blip(test_loader_text, model, device, args)
        image_embeds=BLIP.get_image_embeds_blip(test_loader_image, model, device, args)
    elif config['backbone']=='clip':
        text_embeds=CLIP.get_text_embeds_clip(test_loader_text, model, device, args)
        image_embeds=CLIP.get_image_embeds_clip(test_loader_image, model, device, args)
    elif config['backbone']=='clip_reid':
        text_embeds=CLIP_Reid.get_text_embeds_clip_reid(test_loader_text, model, device, args)
        image_embeds=CLIP_Reid.get_image_embeds_clip_reid(test_loader_image, model, device, args)

    #blip_score
    sims_matrix = F.normalize(image_embeds) @ F.normalize(text_embeds).t()
    score_matrix_i2t = torch.full((len(test_loader_image.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)


    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, header)): 
        score_matrix_i2t[start+i] = sims
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(test_loader_image.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    


    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, header)): 
        score_matrix_t2i[start+i] = sims

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)   

    evaluate_distribution=True
    with torch.no_grad():
        if evaluate_distribution:
            image_embeds=F.normalize(image_embeds)
            text_embeds=F.normalize(text_embeds)
            modality_gap=compute_modality_gap(image_embeds,text_embeds)     
            print("Modality gap", modality_gap.item())
            
            image_center=image_embeds.mean(0)
            image_intra_sim = torch.norm(image_embeds - image_center, dim=1)
            image_intra_sim=image_intra_sim.mean()

            text_center=text_embeds.mean(0)
            text_intra_sim = torch.norm(text_embeds - text_center, dim=1)
            text_intra_sim=text_intra_sim.mean()

            inter_sim=(image_embeds@text_embeds.t()).mean()

            print(f"Image Intra Uniformity: {image_intra_sim:.3f}")
            print(f"Text Intra Uniformity: {text_intra_sim:.3f}")
            print(f"Inter Sim: {inter_sim:.3f}")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, args, data_loader, idx_trans=None):
    if args.retrieval=='i2t':
        txt2img=data_loader.dataset.txt2img
        img2txt = [data_loader.dataset.img2txt[idx] for idx in idx_trans]
    elif args.retrieval=='t2i':
        img2txt=data_loader.dataset.img2txt
        txt2img = [data_loader.dataset.txt2img[idx] for idx in idx_trans]
    else:
        img2txt=data_loader.dataset.img2txt
        txt2img=data_loader.dataset.txt2img

    # Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    average_precisions_i2t = []
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        relevant_indices = img2txt[index]
        relevant_positions = []
        for i in relevant_indices:
            tmp = np.where(inds == i)[0][0]
            relevant_positions.append(tmp)
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

        # Calculate average precision
        relevant_positions.sort()
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
        average_precision = np.mean(precisions) if precisions else 0
        average_precisions_i2t.append(average_precision)

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    # Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    average_precisions_t2i = []


    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        relevant_indices = txt2img[index]
        relevant_positions = []
        for i in relevant_indices:
            tmp = np.where(inds == i)[0][0]
            relevant_positions.append(tmp)
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

        # Calculate average precision
        relevant_positions.sort()
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
        average_precision = np.mean(precisions) if precisions else 0
        average_precisions_t2i.append(average_precision)

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = round((tr1 + tr5 + tr10) / 3, 2)
    ir_mean = round((ir1 + ir5 + ir10) / 3, 2)
    r_mean = round((tr_mean + ir_mean) / 2, 2)

    # Calculate R_sum
    r_sum = tr1 + tr5 + tr10 + ir1 + ir5 + ir10

    # Calculate mAP and format to two decimal places
    mAP_i2t = round(np.mean(average_precisions_i2t) * 100, 2)
    mAP_t2i = round(np.mean(average_precisions_t2i) * 100, 2)

    eval_result =  {
        'txt_r1': tr1,
        'txt_r5': tr5,
        'txt_r10': tr10,
        'txt_r_mean': tr_mean,
        'txt_map': mAP_i2t,
        'img_r1': ir1,
        'img_r5': ir5,
        'img_r10': ir10,
        'img_r_mean': ir_mean,
        'img_map': mAP_t2i,
        'r_mean': r_mean,
        'r_sum': r_sum
    }
    return eval_result


@torch.no_grad()
def itm_eval_and_plot(scores_i2t, scores_t2i, args, data_loader, config, idx_trans=None):
    """
    在原始逻辑基础上增加了数据收集和绘图调用。
    """
    # -----------------------------------------------------
    # 准备 Ground Truth
    # -----------------------------------------------------
    if args.retrieval == 'i2t':
        txt2img = data_loader.dataset.txt2img
        img2txt = [data_loader.dataset.img2txt[idx] for idx in idx_trans]
    elif args.retrieval == 't2i':
        img2txt = data_loader.dataset.img2txt
        txt2img = [data_loader.dataset.txt2img[idx] for idx in idx_trans]
    else:
        img2txt = data_loader.dataset.img2txt
        txt2img = data_loader.dataset.txt2img

    # 用于绘图的数据容器
    tr_plot_confidences = []
    tr_plot_is_correct = []
    ir_plot_confidences = []
    ir_plot_is_correct = []

    # 确保 args 中有 temperature，否则默认 0.07 (CLIP/MoCo常用值)
    temperature = getattr(args, 'temperature', 0.07)

    # -----------------------------------------------------
    # 1. Image -> Text Evaluation
    # -----------------------------------------------------
    ranks = np.zeros(scores_i2t.shape[0])
    average_precisions_i2t = []

    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        relevant_indices = img2txt[index]
        relevant_positions = []
        for i in relevant_indices:
            tmp = np.where(inds == i)[0][0]
            relevant_positions.append(tmp)
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

        # Calculate average precision
        relevant_positions.sort()
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
        average_precision = np.mean(precisions) if precisions else 0
        average_precisions_i2t.append(average_precision)

        # =====================================================
        # [新增] 收集 i2t 数据用于绘图 (仅当当前模式支持时)
        # =====================================================
        if args.retrieval == 'i2t' or args.retrieval == None:
            # 1. 获取置信度 (Softmax Top-1)
            # 将numpy score转tensor计算softmax
            prob = F.softmax(torch.from_numpy(score) / temperature, dim=0)#
            # prob = np.argsort(score)[::-1]
            top1_conf = prob[inds[0]].detach().cpu().numpy()
            tr_plot_confidences.append(top1_conf)
            tmp_iscorrect = 1 if rank == 0 else 0
            tr_plot_is_correct.append(tmp_iscorrect)
            # print(index, top1_conf, tmp_iscorrect)
    # Compute metrics i2t
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # -----------------------------------------------------
    # 2. Text -> Images Evaluation
    # -----------------------------------------------------
    ranks = np.zeros(scores_t2i.shape[0])
    average_precisions_t2i = []

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        relevant_indices = txt2img[index]
        relevant_positions = []
        for i in relevant_indices:
            tmp = np.where(inds == i)[0][0]
            relevant_positions.append(tmp)
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

        # Calculate average precision
        relevant_positions.sort()
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
        average_precision = np.mean(precisions) if precisions else 0
        average_precisions_t2i.append(average_precision)

        # =====================================================
        # [新增] 收集 t2i 数据用于绘图
        # =====================================================
        if args.retrieval == 't2i' or args.retrieval == None:
            # prob = np.argsort(score)[::-1]
            prob = F.softmax(torch.from_numpy(score) / temperature, dim=0)
            top1_conf = prob[inds[0]].detach().cpu().numpy()
            ir_plot_confidences.append(top1_conf)
            tmp_iscorrect = 1 if rank == 0 else 0
            ir_plot_is_correct.append(tmp_iscorrect)
            # print(index, top1_conf, tmp_iscorrect)

    # Compute metrics t2i
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = round((tr1 + tr5 + tr10) / 3, 2)
    ir_mean = round((ir1 + ir5 + ir10) / 3, 2)
    r_mean = round((tr_mean + ir_mean) / 2, 2)
    r_sum = tr1 + tr5 + tr10 + ir1 + ir5 + ir10
    mAP_i2t = round(np.mean(average_precisions_i2t) * 100, 2)
    mAP_t2i = round(np.mean(average_precisions_t2i) * 100, 2)

    eval_result = {
        'txt_r1': tr1,
        'txt_r5': tr5,
        'txt_r10': tr10,
        'txt_r_mean': tr_mean,
        'txt_map': mAP_i2t,
        'img_r1': ir1,
        'img_r5': ir5,
        'img_r10': ir10,
        'img_r_mean': ir_mean,
        'img_map': mAP_t2i,
        'r_mean': r_mean,
        'r_sum': r_sum
    }

    # -----------------------------------------------------
    # 3. 调用绘图
    # -----------------------------------------------------
    print('绘图')
    save_fig_pth = os.path.join(args.config.replace('configs', 'figures').replace('.yaml', ''), args.method)
    os.makedirs(save_fig_pth, exist_ok=True)
    if len(ir_plot_confidences) > 0:
        print(f"Plotting Calibration Metrics for mode: t2i")
        conf_np = np.array(ir_plot_confidences)
        correct_np = np.array(ir_plot_is_correct)

        # 调用绘图函数
        reliability_plot(conf_np, correct_np, num_bins=20, save_fig_pth=save_fig_pth, title_suffix=f"{config['dataset']} t2i")
        plot_confidence_distribution(conf_np, correct_np.astype(bool), n_bins=40, save_fig_pth=save_fig_pth, title_suffix=f"{config['dataset']} t2i")
    if len(tr_plot_confidences) > 0:
        print(f"Plotting Calibration Metrics for mode: i2t")
        conf_np = np.array(tr_plot_confidences)
        correct_np = np.array(tr_plot_is_correct)

        # 调用绘图函数
        reliability_plot(conf_np, correct_np, num_bins=20, save_fig_pth=save_fig_pth, title_suffix=f"{config['dataset']} i2t")
        plot_confidence_distribution(conf_np, correct_np.astype(bool), n_bins=40, save_fig_pth=save_fig_pth, title_suffix=f"{config['dataset']} i2t")
    return eval_result

@torch.no_grad()
def itm_eval_and_plot_2(pred_gap, target_gap, args, config):
    print(len(pred_gap))
    print(len(target_gap))
    assert len(pred_gap) == len(target_gap)

    num_batches = len(pred_gap)
    batches = np.arange(1, num_batches+1)

    plt.figure()

    # 柱状图：预测 gap#BE2418,
    plt.bar(batches, pred_gap, color='#D9D9D9',width=0.8, alpha=0.9,  label="Predicted Gap")

    # 折线图：目标 gap
    plt.plot(batches, target_gap, color="#0000FF", marker='^', markersize=6, linestyle='--', linewidth=1.0, label="Target Gap")

    plt.xlabel("Batch Index")
    plt.ylabel("Top1 - Top2 Gap")
    plt.title("Top1–Top2 Gap per Batch (One Epoch)")
    plt.ylim(0.05, 0.3)
    plt.legend()
    save_fig_pth = os.path.join(args.config.replace('configs', 'figures').replace('.yaml', ''), args.method)
    os.makedirs(save_fig_pth, exist_ok=True)
    title_suffix = f"{config['dataset']} {args.retrieval}"
    plt.savefig(os.path.join(save_fig_pth, title_suffix + ' confidence_gap_diagram.png'))



@torch.no_grad()
def itm_eval_and_plot_3(pred_gap, target_gap, args, config):
    print(len(pred_gap))
    print(len(target_gap))
    assert len(pred_gap) == len(target_gap)

    num_batches = len(pred_gap)
    batches = np.arange(1, num_batches+1)

    plt.figure()

    # 柱状图：预测 gap#BE2418,
    plt.bar(batches, pred_gap, color='#D9D9D9',width=0.8, alpha=0.9,  label="Predicted Gap")

    # 折线图：目标 gap
    plt.plot(batches, target_gap, color="#0000FF", marker='^', markersize=6, linestyle='--', linewidth=1.0, label="Target Gap")

    plt.xlabel("Batch Index")
    plt.ylabel("Top1 - Top2 Gap")
    plt.title("Top1–Top2 Gap per Batch (One Epoch)")
    plt.ylim(0.05, 0.3)
    plt.legend()
    save_fig_pth = os.path.join(args.config.replace('configs', 'figures').replace('.yaml', ''), args.method)
    os.makedirs(save_fig_pth, exist_ok=True)
    title_suffix = f"{config['dataset']} {args.retrieval}"
    plt.savefig(os.path.join(save_fig_pth, title_suffix + ' confidence_gap_diagram.png'))

import os
import numpy as np
import matplotlib.pyplot as plt

def _normalize(x, mode="minmax", eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    if mode is None:
        return x
    if mode == "minmax":
        mn, mx = np.min(x), np.max(x)
        return (x - mn) / (mx - mn + eps)
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"Unknown normalize mode: {mode}")

def normalize_joint_quantile_minmax(predicted_all, expected_source, q_low=0.01, q_high=0.99, eps=1e-12):
    pred = np.asarray(predicted_all, dtype=np.float64)
    exp  = np.asarray(expected_source, dtype=np.float64)

    joint = np.concatenate([pred.ravel(), exp.ravel()], axis=0)
    lo = np.quantile(joint, q_low)
    hi = np.quantile(joint, q_high)

    # clip 到分位数范围，再 minmax
    pred_c = np.clip(pred, lo, hi)
    exp_c  = np.clip(exp,  lo, hi)

    pred_n = (pred_c - lo) / (hi - lo + eps)
    exp_n  = (exp_c  - lo) / (hi - lo + eps)
    return pred_n, exp_n


def plot_alignment_diagram(
    predicted_all,
    expected_source,
    n_bins=15,
    save_fig_pth="./figures",
    title_suffix="",
    args=None,
    normalize="minmax",          # None / "minmax" / "sigmoid"
    binning="predicted",         # "predicted" or "expected" (通常用 predicted)
    weighted_by="all",           # "all" or "source" for error weighting
):
    """
    类 Reliability Diagram：对比 predicted(全体) vs expected(source-like)
    - 分箱依据：binning="predicted" 表示按 predicted_all 分箱
    - 每个 bin 内：画 predicted 均值(蓝) & expected 均值(灰)
    - 误差指标：Alignment Error = sum_bin |pred_mean - exp_mean| * bin_prop
    """
    print(predicted_all)
    print(expected_source)
    predicted_all = np.asarray(predicted_all, dtype=np.float64)
    expected_source = np.asarray(expected_source, dtype=np.float64)
    
    # 1) 归一化（强烈建议）
    #pred, exp = normalize_joint_quantile_minmax(predicted_all, expected_source)
    # pred = _normalize(predicted_all, normalize)
    # exp  = _normalize(expected_source, normalize)
    pred, exp = predicted_all, expected_source

    # 2) 选择分箱依据
    if binning == "predicted":
        bin_base = pred
    elif binning == "expected":
        bin_base = exp
    else:
        raise ValueError("binning must be 'predicted' or 'expected'")

    # 3) 分箱边界（0~1）
    
    bin_boundaries = np.linspace(min(bin_base), max(bin_base), n_bins + 1)
    print('分箱边界',min(bin_base), max(bin_base),bin_boundaries)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # 4) 逐 bin 统计
    bin_pred_means = []
    bin_exp_means = []
    bin_prop_all = []
    bin_prop_src = []

    for lo, hi in zip(bin_lowers, bin_uppers):
        # all-samples in bin (按 chosen bin_base)
        if binning == "predicted":
            in_bin_all = (pred > lo) & (pred <= hi)
            in_bin_src = (exp  > lo) & (exp  <= hi)  # source 也按同一套区间落 bin（常用）
        else:
            in_bin_all = (pred > lo) & (pred <= hi)  # all 依然用 pred 来取均值
            in_bin_src = (exp  > lo) & (exp  <= hi)

        prop_all = in_bin_all.mean() if pred.size > 0 else 0.0
        prop_src = in_bin_src.mean() if exp.size > 0 else 0.0

        if in_bin_all.any():
            pred_mean = pred[in_bin_all].mean()
        else:
            pred_mean = 0.0

        if in_bin_src.any():
            exp_mean = exp[in_bin_src].mean()
        else:
            exp_mean = 0.0

        bin_pred_means.append(pred_mean)
        bin_exp_means.append(exp_mean)
        bin_prop_all.append(prop_all)
        bin_prop_src.append(prop_src)

    bin_pred_means = np.array(bin_pred_means)
    bin_exp_means  = np.array(bin_exp_means)
    bin_prop_all   = np.array(bin_prop_all)
    bin_prop_src   = np.array(bin_prop_src)

    # 5) 类 ECE / alignment error
    if weighted_by == "all":
        w = bin_prop_all
    elif weighted_by == "source":
        w = bin_prop_src
    else:
        raise ValueError("weighted_by must be 'all' or 'source'")

    alignment_error = np.sum(np.abs(bin_pred_means - bin_exp_means) * w)

    # 6) 画图
    os.makedirs(save_fig_pth, exist_ok=True)
    plt.figure(figsize=(7, 6))

    positions = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    step = max(bin_base) / n_bins
    bar_width = step * 0.5

    # Outputs: predicted mean per bin (蓝底)
    plt.bar(
        positions, bin_pred_means, width=bar_width,
        color="#0000FF", alpha=1.0, label="UCA",
        align="center", linewidth=1, zorder=1
    )

    # Expected: expected mean per bin (灰盖)
    plt.bar(
        positions, bin_exp_means, width=bar_width,
        color="#BFBFBF", alpha=0.5, label="Expected (source-like)",
        align="center", linewidth=1, zorder=2
    )

    # Perfect match line：y=x（表示 predicted==expected 的理想对齐）
    plt.plot([0, 1], [0, 1], linestyle="--", color="black",
             linewidth=1.5, label="Perfect Match (y=x)", zorder=3)

    plt.xlim(0,  max(bin_base))
    plt.ylim(0,  max(bin_base)+0.1)
    plt.xlabel("confidence bin", fontsize=14)
    plt.ylabel("similarity gap (normalized)", fontsize=14)
    temp = args.config.split('/')[-1].split('.')[0]
    if 'retrieval' in temp:
        title_suffix = 'B2'+title_suffix
    elif '5' in  temp:
        title_suffix = 'image corruption on '+title_suffix
    elif '7' in  temp:
        title_suffix = 'text corruption on '+title_suffix
    plt.title(
        f"Alignment Diagram {title_suffix}\n"
        f"AlignmentError = {alignment_error:.4f}",
        fontsize=14
    )
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(save_fig_pth, f"{title_suffix} {args.method} alignment_diagram.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return alignment_error, out_path
