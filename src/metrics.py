import numpy as np

def hit_rate(recommended_list, bought_list):
    flags = np.isin(bought_list, recommended_list)

    return (flags.sum() > 0) * 1  


def hit_rate_at_k(recommended_list, bought_list, k=5):   
    return hit_rate(recommended_list[:k], bought_list)


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]

    mask = np.isin(recommended_list, bought_list)
    spent_money = mask * prices_recommended
    recommended_money = (recommended_list > 0) * prices_recommended
    
    return np.sum(spent_money)/np.sum(recommended_money)


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def recall_at_k(recommended_list, bought_list, k=5):

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)

    return recall

def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]

    relevant_mask = np.isin(bought_list, recommended_list)
    revenue_relevant = relevant_mask * prices_bought
    
    return np.sum(revenue_relevant)/np.sum(prices_bought)


def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):
        
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
            
    result = sum_ / sum(flags)
    
    return result


def reciprocal_rank(recommended_list, bought_list):
    flags = np.isin(bought_list, recommended_list)
    indeces = np.where(flags == True)
    
    if(len(indeces[0]) == 0):
        return 0
    
    return 1/(indeces[0][0]+1)
