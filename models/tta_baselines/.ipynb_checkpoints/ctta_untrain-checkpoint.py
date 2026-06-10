import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from models.tta_baselines.param import load_model_and_optimizer, copy_model_and_optimizer

class CTTA_untrain(nn.Module):
    """TCR adapts a model by entropy minimization during testing.

    Once TCRed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "TCR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, modality_query, device, args, metric_logger, queue_list, max_queue_size, update_signal):
        if self.episodic:
            self.reset()

        for step in range(self.steps):
            queue_list, outputs, confidence_gap_all, margin_conf = forward_and_adapt(modality_query, device, args, metric_logger, queue_list, max_queue_size, update_signal, step, self.model, self.optimizer)

        return queue_list, outputs, confidence_gap_all, margin_conf

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.no_grad()
def forward_and_adapt(modality_query, device, args, metric_logger, queue_list, max_queue_size, update_signal, step, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # print('ablation')
    loss_REM, loss_UNI, loss_EMG, loss_CON, queue_list, outputs, target_modality_gap, confidence_gap_all, distances, margin_conf =model.module.forward_ctta_untrain(modality_query, device, queue_list, max_queue_size, update_signal, step, args)
    # loss = 0.0 #args.lambda_rem * loss_REM   #
    # loss=args.lambda_rem*loss_REM  + loss_CON + args.lambda_emg*loss_EMG+ args.lambda_uni*loss_UNI##
    # #loss=args.lambda_rem*loss_REM  + loss_CON + args.lambda_emg*loss_EMG+ args.lambda_uni*loss_UNI##
    # # loss =  loss_REM  +  loss_UNI + loss_EMG+  loss_CON
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    metric_logger.update(loss_REM=loss_REM.item())
    if loss_CON == 0.:
        metric_logger.update(loss_CON=loss_CON)
    else:
        metric_logger.update(loss_CON=loss_CON.item())
    metric_logger.update(loss_UNI=loss_UNI.item())
    metric_logger.update(loss_EMG=loss_EMG.item())
    metric_logger.update(target_modality_gap=target_modality_gap.item())
    #metric_logger.update(confidence_gap=confidence_gap.item())
    metric_logger.update(distances=distances.item())
    #metric_logger.update(loss_total=loss.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return queue_list, outputs, confidence_gap_all, margin_conf
