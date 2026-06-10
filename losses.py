import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1))

def entropy_loss_against_noisy(outputs, entropy_queue, eps=1e-3, if_return_weight=False):
    entropys=softmax_entropy(outputs).sum(1)
    entropy_threshold=entropy_queue.max()
    weight=torch.clamp(torch.tensor(1.0) - entropys.clone().detach() /(entropy_threshold + eps), min=0) #0.1,2
    loss=entropys.mul(weight)#[entropys<=entropy_threshold]
    if if_return_weight:
        return loss.mean(0), weight
    else:
        return loss.mean(0)

def forward_REM(outputs, energy_queue, margin, target_modality_gap, eps=1e-3, if_return_weight=False):
    entropys=softmax_entropy(outputs).sum(1)
    entropy_threshold=energy_queue.max()
    # print("weight", margin.shape, target_modality_gap.shape)
    # weight=torch.clamp(torch.tensor(1.0) - target_modality_gap.clone().detach() /(margin ), min=0) #0.1,2
    weight = torch.clamp( margin / (target_modality_gap.clone().detach()+ eps), max=1)
    # print("weight", weight)
    loss=entropys.mul(weight)#[entropys<=entropy_threshold]
    if if_return_weight:
        return loss.mean(0), weight
    else:
        return loss.mean(0)


def compute_modality_gap(all_image_embeds,all_text_embeds):
    all_image_embeds=F.normalize(all_image_embeds)
    all_text_embeds=F.normalize(all_text_embeds)
    image_embed=all_image_embeds.mean(dim=0)
    text_embed=all_text_embeds.mean(dim=0)
    modality_shift = image_embed - text_embed
    modality_gap = torch.norm(modality_shift, p=2)
    return modality_gap

def compute_modality_gap_ctta(all_image_embeds,all_text_embeds):
    all_image_embeds=F.normalize(all_image_embeds)
    all_text_embeds=F.normalize(all_text_embeds)
    image_embed=all_image_embeds.mean(dim=1)
    text_embed=all_text_embeds.mean(dim=1)
    modality_shift = image_embed - text_embed
    modality_gap = torch.norm(modality_shift, p=2)
    return modality_gap,modality_shift


@torch.no_grad()
def get_current_value_ctta(queue_list, real_interval):
    _, modality_1_embeds, modality_2_embeds, entropys = zip(*queue_list)
    feat_1_queue = torch.stack(modality_1_embeds)
    feat_2_queue = torch.stack(modality_2_embeds)
    current_margin=compute_modality_gap(feat_1_queue, feat_2_queue)
    entropys_queue=torch.stack(entropys)
    outputs = (feat_1_queue @ feat_2_queue.t())
    rank_sim, rank_indices = torch.sort(outputs, descending=True)
    if real_interval == 1:
        # print(1)
        out_real = rank_sim[:, 0]
    else:
        # print(2)
        out_real = rank_sim[:, :real_interval].mean(dim=1)
    # print()
    out_fake = rank_sim[:, 1]  # .mean(dim=1)
    margin_conf = (out_real-out_fake).mean(dim=0)
    return current_margin, entropys_queue,margin_conf
@torch.no_grad()
def get_current_value_ctta5(queue_list, real_interval):
    _, modality_1_embeds, modality_2_embeds, entropys = zip(*queue_list)
    feat_1_queue = torch.stack(modality_1_embeds)
    feat_2_queue = torch.stack(modality_2_embeds)
    current_margin=compute_modality_gap(feat_1_queue, feat_2_queue)
    entropys_queue=torch.stack(entropys)
    outputs = (feat_1_queue @ feat_2_queue.t())
    rank_sim, rank_indices = torch.sort(outputs, descending=True)
    if real_interval == 1:
        # print(1)
        out_real = rank_sim[:, 0]
    else:
        # print(2)
        out_real = rank_sim[:, :real_interval].mean(dim=1)
    # print()
    out_fake = rank_sim[:, 1]  # .mean(dim=1)
    margin_conf = out_real-out_fake
    return current_margin, entropys_queue,margin_conf
    
def get_current_value_ctta_conf(queue_list, real_interval):
    _, modality_1_embeds, modality_2_embeds, entropys = zip(*queue_list)
    feat_1_queue = torch.stack(modality_1_embeds)
    feat_2_queue = torch.stack(modality_2_embeds)
    current_margin=compute_modality_gap(feat_1_queue, feat_2_queue)
    entropys_queue=torch.stack(entropys)
    outputs = (feat_1_queue @ feat_2_queue.t())
    rank_sim, rank_indices = torch.sort(outputs, descending=True)
    if real_interval == 1:
        # print(1)
        out_real = rank_sim[:, 0]
    else:
        # print(2)
        out_real = rank_sim[:, :real_interval].mean(dim=1)
    # print()
    out_fake = rank_sim[:, 1]  # .mean(dim=1)
    margin_conf = (out_real-out_fake).mean(dim=0)
    return current_margin, entropys_queue,margin_conf, out_real-out_fake

def get_current_value(queue_list):
    _, modality_1_embeds, modality_2_embeds, entropys = zip(*queue_list)
    feat_1_queue = torch.stack(modality_1_embeds)
    feat_2_queue = torch.stack(modality_2_embeds)
    current_margin=compute_modality_gap(feat_1_queue, feat_2_queue)
    entropys_queue=torch.stack(entropys)
    return current_margin, entropys_queue

@torch.no_grad()
def update_queue(modality_1_feat, modality_2_feat, queue_list, con_ratio, max_queue_size, args):
    num_to_select = int(con_ratio * modality_1_feat.size(0))

    sample_gap=torch.norm(modality_1_feat - modality_2_feat, p=2, dim=1)
    modality_1_center=modality_1_feat.mean(0)
    modality_2_center=modality_2_feat.mean(0)#all_modality_2_feat.mean(0)#modality_2_feat.mean(0)
    sample_1_diversity=torch.norm(modality_1_feat - modality_1_center, p=2, dim=1)
    sample_2_diversity=torch.norm(modality_2_feat - modality_2_center, p=2, dim=1)
    indictor=2*sample_gap-sample_1_diversity-sample_2_diversity

    entropys=softmax_entropy(modality_1_feat@modality_2_feat.t()/args.temperature).sum(1)

    sorted_indices = torch.argsort(indictor)[:num_to_select]

    for i in sorted_indices:
        indictor_item = indictor[i].detach().item()
        modality_1_feat_item = modality_1_feat[i].detach()
        modality_2_feat_item = modality_2_feat[i].detach()
        entropys_item=entropys[i].detach()
        queue_list.append((indictor_item,modality_1_feat_item, modality_2_feat_item,entropys_item))

    # Min Stack Update
    if len(queue_list) >= max_queue_size:
        queue_list = sorted(queue_list, key=lambda x: x[0], reverse=False)
        queue_list = queue_list[:max_queue_size]
    return queue_list

def center_uniform_loss(x, t=0.1):
    center=x.mean(0)
    distances = torch.norm(x - center, dim=1) * t
    loss = (torch.exp(-distances)).mean()
    return loss

def center_uniform_loss_conf(x, t=0.1):
    center=x.mean(0)
    distances = torch.norm(x - center, dim=1) * t
    loss = (torch.exp(-distances)).mean()
    return loss, distances.mean()
    
def observation(image_embeds, text_embeds, prefix='After TTA:'):
    modality_gap = compute_modality_gap(image_embeds, text_embeds)     
    print(f"{prefix} Modality gap:", modality_gap.item())
    
    image_center = image_embeds.mean(0)
    image_intra_sim = torch.norm(image_embeds - image_center, dim=1)
    image_intra_sim = image_intra_sim.mean()

    text_center = text_embeds.mean(0)
    text_intra_sim = torch.norm(text_embeds - text_center, dim=1)
    text_intra_sim = text_intra_sim.mean()

    print(f"{prefix} Image Intra Uniformity: {image_intra_sim:.3f}")
    print(f"{prefix} Text Intra Uniformity: {text_intra_sim:.3f}")