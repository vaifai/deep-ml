def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    m1,m2 = min(x),max(x)
    if(m1 == m2):
        return [0.0] * len(x)
    normal = []
    for i in x:
        i = (i-m1)/(m2-m1)
        normal.append(i)
    return normal