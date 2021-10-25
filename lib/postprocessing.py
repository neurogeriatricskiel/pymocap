import numpy as np

def _compare_events(annotated, predicted):
    """Compare the event timings of the annotated (true, OMC) events
    with the predicted (IMU) events.

    Parameters
    ----------
    annotated : array
        A numpy array with the integer indexes of annotated events.
    predicted : array
        A numpy array with the integer indexes of predicted events.
    """

    ann2pred = np.empty_like(annotated)
    for i in range(len(annotated)):
        ann2pred[i] = np.argmin(np.abs(predicted - annotated[i]))
    
    pred2ann = np.empty_like(predicted)
    for i in range(len(predicted)):
        pred2ann[i] = np.argmin(np.abs(annotated - predicted[i]))
    
    ann2pred_unique = np.unique(ann2pred)
    for i in range(len(ann2pred_unique)):
        indices = np.argwhere(ann2pred == ann2pred_unique[i])[:,0]
        if len(indices) > 1:
            ann2pred[np.setdiff1d(indices, pred2ann[ann2pred_unique[i]])] = -999
    
    pred2ann_unique = np.unique(pred2ann)
    for i in range(len(pred2ann_unique)):
        indices = np.argwhere(pred2ann == pred2ann_unique[i])[:,0]
        if len(indices) > 1:
            pred2ann[np.setdiff1d(indices, ann2pred[pred2ann_unique[i]])] = -999
    
    indices_ann2pred = np.argwhere(ann2pred > -999)[:,0]
    ann2pred[indices_ann2pred[np.argwhere(pred2ann[ann2pred[indices_ann2pred]] == -999)[:,0]]] = -999

    indices_pred2ann = np.argwhere(pred2ann > -999)[:,0]
    pred2ann[indices_pred2ann[np.argwhere(ann2pred[pred2ann[indices_pred2ann]] == -999)[:,0]]] = -999

    init_time_diff = predicted[pred2ann > -999] - annotated[ann2pred > -999]

    indices_ann2pred = ann2pred[ann2pred > -999]
    indices_pred2ann = pred2ann[pred2ann > -999]
    thr = 50
    for ti in range(len(init_time_diff)-1, -1, -1):
        if init_time_diff[ti] > thr:
            ann2pred[indices_pred2ann[ti]] = -999
            pred2ann[indices_ann2pred[ti]] = -999
    
    time_diff = predicted[pred2ann > -999] - annotated[ann2pred > -999]
    return ann2pred, pred2ann, time_diff