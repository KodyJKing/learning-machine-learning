DEFAULT_MODULUS = int(1e+31)

def stringHash(s, modulus=DEFAULT_MODULUS):
    result = 0
    for i in range(len(s)):
        result = ( (result << 5) - result + ord(s[i]) ) % modulus
    return result

def modelString(model):
    result = []
    model.summary( print_fn= lambda x: result.append(x) )
    return "\n".join(result)
