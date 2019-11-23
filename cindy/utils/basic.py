import edit_distance as ed
import time
import math

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))

def cindy_gradient_clip(model, max_norm=10):
    total_norm = 0;

    # gradient_clip should place after loss.backward()
    # max_norm = 10;
    norm_type= 2
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # if(iter > 100):
    #     pdb.set_trace()
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            # print(p.size(), param_norm)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    if total_norm > max_norm:
        print('total_norm: ', total_norm)

    return total_norm    

def cal_distance(label_list, pre_list):
    y = ed.SequenceMatcher(a = label_list, b = pre_list)
    yy = y.get_opcodes()
    insert = 0
    delete = 0
    replace = 0
    for item in yy:
        if item[0] == 'insert':
            insert += item[-1]-item[-2]
        if item[0] == 'delete':
            delete += item[2]-item[1]
        if item[0] == 'replace':
            replace += item[-1]-item[-2]  
    distance = insert+delete+replace     
    return distance, (delete, replace, insert)  