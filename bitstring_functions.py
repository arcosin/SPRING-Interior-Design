import torch

def bitstring_to_number(vocab, bitstring, rang = (0, 1000)):
    low, high = rang
    val = low + ((high - low) // 2)
    for b in bitstring:
        if b == vocab["left"]:
            high = val
            val = low + ((high - low) // 2)
        elif b == vocab["right"]:
            low = val
            val = low + ((high - low) // 2)
        elif b == vocab["end"]:
            break
    return val

def bitstring_to_number_partial(vocab, bitstring, rang = (0, 1000)):
    low, high = rang
    val = low + ((high - low) // 2)
    for b in bitstring:
        if b == vocab["left"]:
            high = val
            val = low + ((high - low) // 2)
        elif b == vocab["right"]:
            if high - low == 1:
                low = high
                val = high
            else:
                low = val
                val = low + ((high - low) // 2)
        elif b == vocab["end"]:
            val = low + ((high - low) // 2)
            low = val
            high = val
            break
    return (low, high)

def number_to_bitstring(vocab, n, rang = (0, 1000), max_cycles = 30):
    low, high = rang
    val = low + ((high - low) // 2)
    bitstring = [vocab["start"]]
    i = 0
    while val != n and i < max_cycles:
        if val > n:
            bitstring.append(vocab["left"])
            high = val
            val = low + ((high - low) // 2)
        elif val < n:
            bitstring.append(vocab["right"])
            low = val
            val = low + ((high - low) // 2)
        i += 1
    bitstring.append(vocab["end"])
    return bitstring

def pad_batch(bs_list, pad_tok = -1):
    max_size = -1
    for bs in bs_list:
        if len(bs) > max_size:
            max_size = len(bs)
    for bs in bs_list:
        while len(bs) < max_size:
            bs.append(pad_tok)
    return bs_list

def batch_bitstring_to_number(vocab, bitstrings, rang = (0, 1000)):
    nums = []
    for i in range(bitstrings.size(1)):    # Iterate through batch size.
        bitstring = bitstrings[:, i:i+1, :]
        n = bitstring_to_number(vocab, bitstring, rang = rang)
        nums.append([n])
    return torch.LongTensor(nums)   # Size: torch.Size([bs, 1]).

def batch_number_to_bitstring(vocab, ns, rang = (0, 1000), max_cycles = 30, pad_tok = -1):
    bitstrings = []
    for i in range(ns.size(0)):    # Iterate through batch size.
        n = ns[i:i+1, :]
        bitstring = number_to_bitstring(vocab, n, rang = rang, max_cycles = max_cycles)
        bitstrings.append(bitstring)
    bitstrings = pad_batch(bitstrings, pad_tok = pad_tok)
    bitstrings = torch.LongTensor(bitstrings)
    return bitstrings.permute(1, 0).unsqueeze(-1)   # Size: torch.Size([padded_length, bs, 1]).


#-------------------------------------------------------------------------------

def main():
    import random
    print("Testing bitstring functions.")
    vocab = {"left": 0, "right": 1, "end": 2, "start": 3}
    nums = [[random.randint(0, 1000)] for _ in range(16)]
    print("Nums:")
    print(nums)
    nums = torch.LongTensor(nums)
    print("Nums shape:  ", nums.size())
    bitstrings = batch_number_to_bitstring(vocab, nums, pad_tok = 4)
    print("Bitstrings:")
    bitstrings_iter = bitstrings.clone().squeeze(-1).permute(1,0)
    for b in bitstrings_iter:
        print(b.tolist())
    print("Bitstrings shape:  ", bitstrings.size())
    nums = batch_bitstring_to_number(vocab, bitstrings)
    print("Nums (pivoted):")
    print(nums.tolist())
    nums = torch.LongTensor(nums)
    print("Nums (pivoted) shape:  ", nums.size())
    print("\n\nTesting done.")


if __name__ == '__main__':
    main()

#===============================================================================
