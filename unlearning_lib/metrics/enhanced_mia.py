import numpy as np

def from_correct_logit_to_loss(array): # convert correct logit to the cross entropy loss
    # return np.log((1+np.exp(array))/np.exp(array)) # positive
    # This computes log(1 + exp(-array)) stably
    return np.logaddexp(0, -array)

def enhanced_mia(target_unl_logit, shadow_target_ori_logit):


    losses = from_correct_logit_to_loss(shadow_target_ori_logit).numpy() # previous shape nb_models x nb_target, ref lossses
    check_losses = from_correct_logit_to_loss(target_unl_logit).numpy() # previous shape nb_target x 1, target losses

    dummy_min = np.zeros((1, len(losses[0]))) # shape 1 x nb_target

    dummy_max = dummy_min + 1000 # shape 1 x nb_target

    dat_reference_or_distill = np.sort(np.concatenate((losses, dummy_max, dummy_min), axis=0), axis=0) # shape nb_models + 2 x nb_target 

    prediction = np.array([])

    discrete_alpha = np.linspace(0, 1, len(dat_reference_or_distill))
    for i in range(len(dat_reference_or_distill[0])):
        losses_i =  dat_reference_or_distill[:, i]

        # Create the interpolator
        pr = np.interp(check_losses[i], losses_i, discrete_alpha)
        prediction = np.append(prediction, pr)

    return prediction

def enhanced_mia_p(target_unl_logit, pop_unl_logit):
    check_losses = from_correct_logit_to_loss(target_unl_logit) # previous shape nb_target x 1, target losses
    pop_unl_losses = from_correct_logit_to_loss(pop_unl_logit)

    prediction = []
    answers = []
    ref = pop_unl_losses.reshape(-1, 1).repeat_interleave(len(check_losses), axis=1) # to parallelize the computation of cdf

    cdf_out = ((ref < check_losses )*1.0).mean(0)
    score = cdf_out

    prediction = np.array(score)
    return prediction