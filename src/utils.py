from torch import nn
import torch
import numpy as np
import os
import copy
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_calibration_error


def read_data(dataset, idx, is_train=True):
    dataset_path = 'dataset'
    if is_train:
        train_data_dir = os.path.join(dataset_path, dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(dataset_path, dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data



def select_read(dataset, select_list):
    imgs_train = []
    labels_train = []
    for i in select_list:
        list = read_data(dataset, i, True)
        imgs_train.extend(list['x'])
        labels_train.extend(list['y'])
    X_train = torch.Tensor(np.array(imgs_train)).type(torch.float32)
    y_train = torch.Tensor(np.array(labels_train)).type(torch.int64)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]

    imgs_test = []
    labels_test = []
    sample_num = 0
    for i in select_list:
        list = read_data(dataset, i, False)
        print(i)
        sample_num += len(list["x"])
        print(f'num_sample: {sample_num}')
        imgs_test.extend(list['x'])
        labels_test.extend(list['y'])
    X_test = torch.Tensor(np.array(imgs_test)).type(torch.float32)
    y_test = torch.Tensor(np.array(labels_test)).type(torch.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]

    return train_data, test_data, sample_num






class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out




def get_bandwidth(f, device):
    """
    Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param device: The device type: 'cpu' or 'cuda'

    :return: The bandwidth of the kernel
    """
    bandwidths = torch.cat((torch.logspace(start=-5, end=-1, steps=15), torch.linspace(0.2, 1, steps=5)))
    max_b = -1
    max_l = 0
    n = len(f)
    for b in bandwidths:
        log_kern = get_kernel(f, b, device)
        log_fhat = torch.logsumexp(log_kern, 1) - torch.log((n - 1) * b)
        l = torch.sum(log_fhat)
        if l > max_l:
            max_l = l
            max_b = b

    return max_b

def get_ece_kde(f, y, bandwidth, p, mc_type, device):
    """
    Calculate an estimate of Lp calibration error.

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param y: The vector containing the labels, shape [num_samples]
    :param bandwidth: The bandwidth of the kernel
    :param p: The p-norm. Typically, p=1 or p=2
    :param mc_type: The type of multiclass calibration: canonical, marginal or top_label
    :param device: The device type: 'cpu' or 'cuda'

    :return: An estimate of Lp calibration error
    """
    check_input(f, bandwidth, mc_type)
    if f.shape[1] == 1:
        return 2 * get_ratio_binary(f, y, bandwidth, p, device)
    else:
        if mc_type == 'canonical':
            return get_ratio_canonical(f, y, bandwidth, p, device)
        elif mc_type == 'marginal':
            return get_ratio_marginal_vect(f, y, bandwidth, p, device)
        elif mc_type == 'top_label':
            return get_ratio_toplabel(f, y, bandwidth, p, device)

def get_ratio_binary(f, y, bandwidth, p, device):
    assert f.shape[1] == 1

    log_kern = get_kernel(f, bandwidth, device)

    return get_kde_for_ece(f, y, log_kern, p)

def get_ratio_canonical(f, y, bandwidth, p, device):
    if f.shape[1] > 60:
        # Slower but more numerically stable implementation for larger number of classes
        return get_ratio_canonical_log(f, y, bandwidth, p, device)

    log_kern = get_kernel(f, bandwidth, device)
    kern = torch.exp(log_kern)

    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    kern_y = torch.matmul(kern, y_onehot)
    den = torch.sum(kern, dim=1)
    # to avoid division by 0
    den = torch.clamp(den, min=1e-10)

    ratio = kern_y / den.unsqueeze(-1)
    ratio = torch.sum(torch.abs(ratio - f) ** p, dim=1)

    return torch.mean(ratio)

# Note for training: Make sure there are at least two examples for every class present in the batch, otherwise
# LogsumexpBackward returns nans.
def get_ratio_canonical_log(f, y, bandwidth, p, device='cpu'):
    log_kern = get_kernel(f, bandwidth, device)
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_y = torch.log(y_onehot)
    log_den = torch.logsumexp(log_kern, dim=1)
    final_ratio = 0
    for k in range(f.shape[1]):
        log_kern_y = log_kern + (torch.ones([f.shape[0], 1]).to(device) * log_y[:, k].unsqueeze(0))
        log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
        inner_ratio = torch.exp(log_inner_ratio)
        inner_diff = torch.abs(inner_ratio - f[:, k]) ** p
        final_ratio += inner_diff

    return torch.mean(final_ratio)

def get_ratio_marginal_vect(f, y, bandwidth, p, device):
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_kern_vect = beta_kernel(f, f, bandwidth).squeeze()
    log_kern_diag = torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)
    # Multiclass case
    log_kern_diag_repeated = f.shape[1] * [log_kern_diag]
    log_kern_diag_repeated = torch.stack(log_kern_diag_repeated, dim=2)
    log_kern_vect = log_kern_vect + log_kern_diag_repeated

    return get_kde_for_ece_vect(f, y_onehot, log_kern_vect, p)

def get_ratio_toplabel(f, y, bandwidth, p, device):
    f_max, indices = torch.max(f, 1)
    f_max = f_max.unsqueeze(-1)
    y_max = (y == indices).to(torch.int)

    return get_ratio_binary(f_max, y_max, bandwidth, p, device)

def get_kde_for_ece_vect(f, y, log_kern, p):
    log_kern_y = log_kern * y
    # Trick: -inf instead of 0 in log space
    log_kern_y[log_kern_y == 0] = torch.finfo(torch.float).min

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f) ** p

    return torch.sum(torch.mean(ratio, dim=0))

def get_kde_for_ece(f, y, log_kern, p):
    f = f.squeeze()
    N = len(f)
    # Select the entries where y = 1
    idx = torch.where(y == 1)[0]
    if not idx.numel():
        return torch.sum((torch.abs(-f)) ** p) / N

    if idx.numel() == 1:
        # because of -inf in the vector
        log_kern = torch.cat((log_kern[:idx], log_kern[idx + 1:]))
        f_one = f[idx]
        f = torch.cat((f[:idx], f[idx + 1:]))

    log_kern_y = torch.index_select(log_kern, 1, idx)

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f) ** p

    if idx.numel() == 1:
        return (ratio.sum() + f_one ** p) / N

    return torch.mean(ratio)

def get_kernel(f, bandwidth, device):
    # if num_classes == 1
    if f.shape[1] == 1:
        log_kern = beta_kernel(f, f, bandwidth).squeeze()
    else:
        log_kern = dirichlet_kernel(f, bandwidth).squeeze()
    # Trick: -inf on the diagonal
    return log_kern + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)

def beta_kernel(z, zi, bandwidth=0.1):
    p = zi / bandwidth + 1
    q = (1 - zi) / bandwidth + 1
    z = z.unsqueeze(-2)

    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    log_num = (p - 1) * torch.log(z) + (q - 1) * torch.log(1 - z)
    log_beta_pdf = log_num - log_beta

    return log_beta_pdf

def dirichlet_kernel(z, bandwidth=0.1):
    alphas = z / bandwidth + 1

    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
    log_num = torch.matmul(torch.log(z), (alphas - 1).T)
    log_dir_pdf = log_num - log_beta

    return log_dir_pdf

def check_input(f, bandwidth, mc_type):
    assert not isnan(f)
    assert len(f.shape) == 2
    assert bandwidth > 0
    assert torch.min(f) >= 0
    assert torch.max(f) <= 1.1

def isnan(a):
    return torch.any(torch.isnan(a))





def cal_ECE(outputs, labels, num_bins):
    confidences = np.max(outputs, 1)
    step = 1.0 / num_bins
    bin_lowers = [i * step for i in range(num_bins)]
    bin_uppers = [(i + 1) * step for i in range(num_bins)]
    predictions = np.argmax(outputs, 1)
    accuracies = predictions == labels

    xs = []
    ys = []
    zs = []
    counts = []
    # ece = Variable(torch.zeros(1)).type_as(confidences)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.sum()
        accuracy_in_bin = accuracies[in_bin].sum()
        avg_confidence_in_bin = confidences[in_bin].sum()
        xs.append(avg_confidence_in_bin)
        ys.append(accuracy_in_bin)
        zs.append(prop_in_bin)
        counts.append(accuracies[in_bin].shape[0])
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    counts = np.array(counts)

    out = {"confidence": xs, "accuracy": ys, "p": zs, "count": counts}

    return out


def calibration_curve(outputs, labels, num_bins, nc=10):
    if outputs is None:
        out = None
    else:
        confidences = np.max(outputs, 1)
        step = (confidences.shape[0] + num_bins - 1) // num_bins
        bins = np.sort(confidences)[::step]
        if confidences.shape[0] % step != 1:
            bins = np.concatenate((bins, [np.max(confidences)]))
        # bins = np.linspace(0.1, 1.0, 30)
        predictions = np.argmax(outputs, 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        accuracies = predictions == labels

        xs = []
        ys = []
        zs = []

        # ece = Variable(torch.zeros(1)).type_as(confidences)
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower) * (confidences < bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                xs.append(avg_confidence_in_bin)
                ys.append(accuracy_in_bin)
                zs.append(prop_in_bin)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        brier = cal_brier(outputs, labels, num_classes=nc)
        NLL = torch.nn.functional.nll_loss(torch.tensor(np.log(outputs + 1e-10)), torch.tensor(labels)).item()

        kde_ece = get_ece_kde(torch.from_numpy(outputs).cuda(), torch.from_numpy(labels).cuda(), bandwidth=0.001, p=1, mc_type='canonical', device='cuda').cpu().item()
        l2_ce = get_ece_kde(torch.from_numpy(outputs).cuda(), torch.from_numpy(labels).cuda(), bandwidth=0.001, p=2, mc_type='canonical', device='cuda').cpu().item()

        out = {"confidence": xs, "accuracy": ys, "p": zs, "ece": ece, "brier": brier, "NLL": NLL,
               "bins": (bin_lowers + bin_uppers) / 2, 'kde_ece': kde_ece, 'l2_ce': l2_ce}

    return out


def evaluate_(model, test_loader, num_classes=10):
    model.eval()
    device = 'cuda'
    list_predictions, list_labels = [], []

    correct, total = 0, 0
    for samples, labels in iter(test_loader):
        samples = samples.to(device)
        labels = labels.to(device)
        # samples: B,C,H,W
        predictions = model(samples)
        predictions = F.softmax(predictions, dim=1)

        _, pred_labels = torch.max(predictions, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        list_predictions.append(predictions.cpu().detach())
        list_labels.append(labels.cpu().detach())
    predictions = torch.cat(list_predictions, dim=0)
    labels = torch.cat(list_labels, dim=0)
    if predictions[0][0] < 0:
        predictions_array = np.exp(predictions.cpu().detach().numpy())  # case in log-softmax
    else:
        predictions_array = predictions.cpu().detach().numpy()
    labels_array = labels.cpu().detach().numpy()
    acc = correct * 1.0 / total
    out = calibration_curve(predictions_array, labels_array, 20, nc=num_classes)

    out_uni = cal_ECE(predictions_array, labels_array, 20)

    out['predictions'] = predictions_array
    out['labels'] = labels_array
    out['acc'] = acc
    out['uni'] = out_uni

    return out


def evaluate_aph(model, test_loader, heads, num_classes=10):
    aph_model = APH(model.base, heads)
    aph_model.eval()
    list_predictions, list_labels = [], []
    correct = 0
    total = 0
    for samples, labels in iter(test_loader):
        samples = samples.to('cuda')
        labels = labels.to('cuda')
        # samples: B,C,H,W
        predictions = aph_model(samples)

        _, pred_labels = torch.max(predictions, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        list_predictions.append(predictions.cpu().detach())
        list_labels.append(labels.cpu().detach())
    predictions = torch.cat(list_predictions, dim=0)
    labels = torch.cat(list_labels, dim=0)
    if predictions[0][0] < 0:
        predictions_array = np.exp(predictions.cpu().detach().numpy())  # case in log-softmax
    else:
        predictions_array = predictions.cpu().detach().numpy()
    labels_array = labels.cpu().detach().numpy()
    acc = correct * 1.0 / total
    out = calibration_curve(predictions_array, labels_array, 20, nc=num_classes)

    out_uni = cal_ECE(predictions_array, labels_array, 20)

    out['predictions'] = predictions_array
    out['labels'] = labels_array
    out['acc'] = acc
    out['uni'] = out_uni

    return out


def aggregate_out(weight, out_list):
    list_acc, list_ece, list_brier, list_nll, list_kde_ece, list_l2_ce = [], [], [], [], [], []
    list_confidence, list_accuracy, list_p, list_count = [], [], [], []
    for i in range(len(out_list)):
        out = out_list[i]
        list_acc.append(out['acc'])
        list_nll.append(out['NLL'])
        list_brier.append(out['brier'])
        list_ece.append(out['ece'])
        list_kde_ece.append(out['kde_ece'])
        list_l2_ce.append(out['l2_ce'])

        list_confidence.append(out['uni']['confidence'])
        list_accuracy.append(out['uni']['accuracy'])
        list_p.append(out['uni']['p'])
        list_count.append(out['uni']['count'])
    arr_conf = np.stack(list_confidence, axis=0)
    arr_accuracy = np.stack(list_accuracy, axis=0)
    arr_p = np.stack(list_p, axis=0)
    arr_count = np.stack(list_count, axis=0)

    arr_count = np.sum(arr_count, axis=0)
    arr_conf = np.sum(arr_conf, axis=0)
    arr_accuracy = np.sum(arr_accuracy, axis=0)
    arr_p = np.sum(arr_p, axis=0)

    flag = arr_count != 0
    arr_conf = arr_conf[flag]
    arr_accuracy = arr_accuracy[flag]
    arr_p = arr_p[flag]
    arr_count = arr_count[flag]

    arr_conf = arr_conf / arr_count
    arr_accuracy = arr_accuracy / arr_count
    arr_p = arr_p / arr_p.sum()

    arr_ece = np.abs(arr_conf - arr_accuracy) * arr_p
    ece_overall = arr_ece.sum()  # ece tested on full dataset

    acc = (np.array(list_acc) * weight).sum()
    nll = (np.array(list_nll) * weight).sum()
    brier = (np.array(list_brier) * weight).sum()
    F_ECE = (np.array(list_ece) * weight).sum()
    F_KDE_ECE = (np.array(list_kde_ece) * weight).sum()
    F_L2_CE = (np.array(list_l2_ce) * weight).sum()


    ece_client = list_ece  # client ece list

    metrics = {'acc': acc,
               'nll': nll,
               'brier': brier,
               'p_ece': F_ECE,
               'g_ece': ece_overall,
               'F_KDE_ECE': F_KDE_ECE,
               'F_L2_CE': F_L2_CE,
               'ece_client': ece_client,
               'acc_client': list_acc,
               'nll_client': list_nll,
               'brier_client': list_brier}
    return metrics


def recover_array(labels, num_classes=10):
    length = labels.shape[0]
    label_array = np.zeros([length, num_classes])
    for i in range(length):
        label_array[i, labels[i]] = 1
    return label_array


def cal_brier(out, labels, num_classes=10):
    label_array = recover_array(labels, num_classes=num_classes)
    instance_num = labels.shape[0]
    brier_score = np.sum((out - label_array) ** 2) / instance_num
    return brier_score


def evaluate_from_head(model, test_loader, heads, num_classes=10):
    model.eval()
    device = 'cuda'
    head_num = len(heads)
    correct, total = 0, 0
    for i in range(head_num):
        list_predictions, list_labels = [], []
        head = heads[i]
        model.head = copy.deepcopy(head)
        model.eval()
        for samples, labels in iter(test_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            # samples: B,C,H,W
            predictions = model(samples)
            predictions = F.softmax(predictions, dim=1)
            list_predictions.append(predictions.cpu().detach())
            list_labels.append(labels.cpu().detach())
        if i == 0:
            predictions_total = (1.0 / head_num) * torch.cat(list_predictions, dim=0)
        else:
            predictions_total += (1.0 / head_num) * torch.cat(list_predictions, dim=0)
        labels = torch.cat(list_labels, dim=0)
    _, pred_labels = torch.max(predictions_total, 1)
    pred_labels = pred_labels.view(-1)
    correct += torch.sum(torch.eq(pred_labels, labels)).item()
    total += len(labels)

    if predictions_total[0][0] < 0:
        predictions_array = np.exp(predictions_total.cpu().detach().numpy())  # case in log-softmax
    else:
        predictions_array = predictions_total.cpu().detach().numpy()
    labels_array = labels.cpu().detach().numpy()
    acc = correct * 1.0 / total
    out = calibration_curve(predictions_array, labels_array, 20, nc=num_classes)

    out_uni = cal_ECE(predictions_array, labels_array, 20)

    out['predictions'] = predictions_array
    out['labels'] = labels_array
    out['acc'] = acc
    out['uni'] = out_uni

    return out


def set_dropout(model):
    for block in model.modules():
        if type(block) == torch.nn.modules.dropout.Dropout:
            block.train()


def evaluate_from_dropout(model, test_loader, num_classes=10, sampling_num=10):
    model.eval()
    set_dropout(model)
    device = 'cuda'
    correct, total = 0, 0
    for i in range(sampling_num):
        list_predictions, list_labels = [], []
        for samples, labels in iter(test_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            # samples: B,C,H,W
            predictions = model(samples)
            predictions = F.softmax(predictions, dim=1)
            list_predictions.append(predictions.cpu().detach())
            list_labels.append(labels.cpu().detach())
        if i == 0:
            predictions_total = (1.0 / sampling_num) * torch.cat(list_predictions, dim=0)
        else:
            predictions_total += (1.0 / sampling_num) * torch.cat(list_predictions, dim=0)
        labels = torch.cat(list_labels, dim=0)
    _, pred_labels = torch.max(predictions_total, 1)
    pred_labels = pred_labels.view(-1)
    correct += torch.sum(torch.eq(pred_labels, labels)).item()
    total += len(labels)

    if predictions_total[0][0] < 0:
        predictions_array = np.exp(predictions_total.cpu().detach().numpy())  # case in log-softmax
    else:
        predictions_array = predictions_total.cpu().detach().numpy()
    labels_array = labels.cpu().detach().numpy()
    acc = correct * 1.0 / total
    out = calibration_curve(predictions_array, labels_array, 20, nc=num_classes)

    out_uni = cal_ECE(predictions_array, labels_array, 20)

    out['predictions'] = predictions_array
    out['labels'] = labels_array
    out['acc'] = acc
    out['uni'] = out_uni

    return out


def evaluate_from_ensemble(model_list, test_loader, num_classes=10):
    device = 'cuda'
    correct, total = 0, 0
    sampling_num = len(model_list)
    for i in range(len(model_list)):
        list_predictions, list_labels = [], []
        model = model_list[i]
        model.eval()
        for samples, labels in iter(test_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            # samples: B,C,H,W
            predictions = model(samples)
            predictions = F.softmax(predictions, dim=1)
            list_predictions.append(predictions.cpu().detach())
            list_labels.append(labels.cpu().detach())
        if i == 0:
            predictions_total = (1.0 / sampling_num) * torch.cat(list_predictions, dim=0)
        else:
            predictions_total += (1.0 / sampling_num) * torch.cat(list_predictions, dim=0)
        labels = torch.cat(list_labels, dim=0)
    _, pred_labels = torch.max(predictions_total, 1)
    pred_labels = pred_labels.view(-1)
    correct += torch.sum(torch.eq(pred_labels, labels)).item()
    total += len(labels)

    if predictions_total[0][0] < 0:
        predictions_array = np.exp(predictions_total.cpu().detach().numpy())  # case in log-softmax
    else:
        predictions_array = predictions_total.cpu().detach().numpy()
    labels_array = labels.cpu().detach().numpy()
    acc = correct * 1.0 / total
    out = calibration_curve(predictions_array, labels_array, 20, nc=num_classes)

    out_uni = cal_ECE(predictions_array, labels_array, 20)

    out['predictions'] = predictions_array
    out['labels'] = labels_array
    out['acc'] = acc
    out['uni'] = out_uni

    return out


class APH(nn.Module):
    def __init__(self, base, head_list):
        super().__init__()
        self.feature_extractor = copy.deepcopy(base)
        self.head_list = copy.deepcopy(head_list)
        self.heads_num = len(self.head_list)
    def forward(self, x):
        self.feature_extractor.eval()
        feature = self.feature_extractor(x)
        predictions_total = None
        i = -1
        for head in self.head_list:
            head.eval()
            i = i + 1
            pre = head(feature)
            pre = F.softmax(pre, dim=1)
            if i == 0:
                predictions_total = (1.0/self.heads_num) * pre
            else:
                predictions_total += (1.0/self.heads_num) * pre
        return predictions_total
