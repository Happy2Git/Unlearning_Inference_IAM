import torch

def carlini_logit(logit, y):
    preds = logit - torch.max(logit, dim=1, keepdim=True)[0]
    preds = torch.exp(preds)
    preds = preds / torch.sum(preds, dim=1, keepdim=True)
    COUNT = preds.shape[0]
    y_true = preds[torch.arange(COUNT), y[:COUNT]]
    preds[torch.arange(COUNT), y[:COUNT]] = 0

    # Sum the probabilities for all wrong preds
    y_wrong = torch.sum(preds, dim=1)
    logit = (torch.log(y_true + 1e-45) - torch.log(y_wrong + 1e-45))
    return logit.cpu()

def model_confidence(logit, y):
    """Computes the model confidence for a given example (x, y)
        Parameters
        ----------
            model : torch model
            x : tensor, List of data for the example we want to perform membership inference on
            y : The labels associated to x

        Returns
        -------
            model_confidence : exp(-CrossEntropyLoss(x, y)) which is in [0, 1]
    """
    preds_softmax = torch.softmax(logit, dim=1)  # Apply softmax to get probabilities
    # Gather the confidence score for the ground truth label of each sample
    confidence_scores = preds_softmax[range(len(y)), y]
    return confidence_scores.cpu()

def model_confidence_with_success(logit, y):
    """Computes the model confidence for a given example (x, y)
        Parameters
        ----------
            model : torch model
            x : tensor, List of data for the example we want to perform membership inference on
            y : The labels associated to x

        Returns
        -------
            model_confidence : exp(-CrossEntropyLoss(x, y)) which is in [0, 1]
    """
    preds_softmax = torch.softmax(logit, dim=1)  # Apply softmax to get probabilities
    # Gather the confidence score for the ground truth label of each sample
    confidence_scores = preds_softmax[range(len(y)), y]
    # get the predicted successful binary results
    predict = torch.argmax(logit, dim=1)
    success = predict == y
    return confidence_scores.cpu(), success.cpu()

def logit_scaling(p):
    """Computes the logit scaling of a given probability so that the model's confidence is approximately normally distributed
        Parameters
        ----------
            p : tensor, probability

        Returns
        -------
            logit_scaling : log(p / (1 - p)) which is in (-inf, inf)
    """
    return torch.log((p + 1e-45) / (1 - p + 1e-45)).cpu()


def predict_proba(logit, x):
    """Computes the model's probability for each class for a given example x
        Parameters
        ----------
            model : torch model
            x : tensor, List of data for the example we want to perform membership inference on

        Returns
        -------
            proba : tensor, List of probabilities for each class which sum to 1
    """
    return torch.softmax(logit, dim=1).detach().cpu()

def ce_loss(logit, y):
    """Computes the cross entropy loss for a given example (x, y)
        Parameters
        ----------
            model : torch model
            x : tensor, List of data for the example we want to perform membership inference on
            y : The labels associated to x

        Returns
        -------
            ce_loss : tensor, Cross entropy loss
    """
    loss = torch.nn.CrossEntropyLoss(reduction="none", reduce=False)
    return loss(logit, y).detach().cpu()