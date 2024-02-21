
def str_number(num):
    if num > 1e14:
        return f"{num/1e12:.0f}T"
    elif num > 1e12:
        return f"{num/1e12:.1f}T"
    elif num>1e11:
        return f"{num/1e9:.0f}G"
    elif num > 1e9:
        return f"{num/1e9:.1f}G"
    elif num > 1e8:
        return f"{num/1e6:.0f}M"
    elif num > 1e6:
        return f"{num/1e6:.1f}M"
    elif num > 1e5:
        return f"{num/1e3:.0f}K"
    elif num > 1e3:
        return f"{num/1e3:.1f}K"
    elif num >= 1:
        return f"{num:.1f}"
    else:
        return f"{num:.2f}"

def str_number_time(num):
    if num >= 1:
        return f"{num:.1f}"
    elif num > 1e-3:
        return f"{num*1e3:.1f}m"
    elif num > 1e-6:
        return f"{num*1e6:.1f}u"
    elif num > 1e-9:
        return f"{num*1e9:.1f}n"
    else:
        return f"{num:.0f}"