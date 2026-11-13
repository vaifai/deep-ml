import numpy as np


def update_ratios(params, grads, lr, low=-4.0, high=-2.0):
    """For every 2-D parameter: {"index", "ratio" = log10(std(lr * grad) / std(param)), "status"}."""
    answer = []
    for i in range(len(params)):
        if(params[i].ndim == 2):
            d = {}
            d["index"] = i
            if(np.std(params[i]) == 0.0):
                d["ratio"] = None
                d["status"] = "undefined"
            else:
                d["ratio"] = round(np.log10(np.std(lr*grads[i])/np.std(params[i])).item(), 3)
                if(d["ratio"] == 0.0):
                    d["status"] = "undefined"
                elif(d["ratio"] < low):
                    d["status"] = "slow"
                elif(d["ratio"] > high):
                    d["status"] = "fast"
                else:
                    d["status"] = "ok"
            answer.append(d)
    return answer



def lr_multiplier(ratios, target=-3.0):
    """The factor to multiply the learning rate by so the median ratio lands on target."""
    sorted_ratios = sorted(x for x in ratios if x is not None)
    n = len(sorted_ratios)
    if n%2 == 1:
        return round(np.power(10,target - sorted_ratios[int(n/2)]).item(), 3)
    else:
        med = (sorted_ratios[int(n/2)] + sorted_ratios[int(n/2) - 1])/2
        return round(np.pow(10, target - med), 3)
    

