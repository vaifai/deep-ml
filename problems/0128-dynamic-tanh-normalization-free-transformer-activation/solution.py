import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    # Your code here
    if(x.ndim == 1):
        return get_dtanh_1d(x, alpha, gamma, beta)
    elif(x.ndim == 2):
        return get_dtanh_2d(x, alpha, gamma, beta)
    return get_dtanh_3d(x, alpha, gamma, beta)

def get_dtanh_1d(arr: np.array, alpha, gamma, beta):
    arr_1d = alpha*arr
    return (gamma*np.tanh(arr_1d) + beta).tolist()
def get_dtanh_2d(arr: np.ndarray, alpha, gamma, beta):
    answer=[]
    for arr_1d in arr:
        answer.append(get_dtanh_1d(arr_1d, alpha, gamma, beta))
    return answer
def get_dtanh_3d(arr: np.ndarray, alpha, gamma, beta):
    answer=[]
    for arr_2d in arr:
        answer.append(get_dtanh_2d(arr_2d, alpha, gamma, beta))
    return answer
