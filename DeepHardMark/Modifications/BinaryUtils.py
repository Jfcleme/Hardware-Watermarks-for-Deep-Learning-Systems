import numpy as np
import torch


def Array2Bin(nums):
    binary_nums = float2bit(nums)
    return binary_nums


def Bin2Array(nums):
    binary_nums = bit2float(nums)
    return binary_nums


def bit2float(b, num_e_bits=8, num_m_bits=23, bias=127.):
    """Turn input tensor into float.
        Args:
            b : binary tensor. The last dimension of this tensor should be the
            the one the binary is at.
            num_e_bits : Number of exponent bits. Default: 8.
            num_m_bits : Number of mantissa bits. Default: 23.
            bias : Exponent bias/ zero offset. Default: 127.
        Returns:
            Tensor: Float tensor. Reduces last dimension.
    """
    expected_last_dim = num_m_bits + num_e_bits + 1
    assert b.shape[-1] == expected_last_dim, "Binary tensors last dimension " \
                                             "should be {}, not {}.".format(
        expected_last_dim, b.shape[-1])

    # check if we got the right type
    dtype = torch.float32
    if expected_last_dim > 32: dtype = torch.float64
    if expected_last_dim > 64:
        print("pytorch can not process floats larger than 64 bits, keep"
                      " this in mind. Your result will be not exact.")
        return

    s = torch.index_select(b, -1, torch.arange(0, 1, device="cuda:0"))
    e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits, device="cuda:0"))
    m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                               1 + num_e_bits + num_m_bits, device="cuda:0"))
    # SIGN BIT
    out = ((-1) ** s).squeeze(-1).type(dtype)
    # EXPONENT BIT
    exponents = -torch.arange(-(num_e_bits - 1.), 1., device="cuda:0")
    exponents = exponents.repeat(b.shape[:-1] + (1,))
    e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
    out *= 2 ** e_decimal
    # MANTISSA
    matissa = (torch.Tensor([2.]) ** (
        -torch.arange(1., num_m_bits + 1.))).repeat(
        m.shape[:-1] + (1,)).cuda()
    out *= 1. + torch.sum(m * matissa, dim=-1)
    return out


def float2bit(f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
    """Turn input tensor into binary.
        Args:
            f : float tensor.
            num_e_bits : Number of exponent bits. Default: 8.
            num_m_bits : Number of mantissa bits. Default: 23.
            bias : Exponent bias/ zero offset. Default: 127.
            dtype : This is the actual type of the tensor that is going to be
            returned. Default: torch.float32.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
    """
    ## SIGN BIT
    s = torch.sign(f)
    s[s==0] = 1
    f = f * s
    # turn sign into sign-bit
    s = (s * (-1) + 1.) * 0.5
    s = s.unsqueeze(-1)

    ## EXPONENT BIT
    e_scientific = torch.clip(torch.floor(torch.log2(f)), -bias, bias+1)
    e_decimal = e_scientific + bias
    e = integer2bit(e_decimal, num_bits=num_e_bits)

    ## MANTISSA
    m1 = integer2bit(f - f % 1, num_bits=num_e_bits)
    m2 = remainder2bit(f % 1, num_bits=bias)
    m3 = torch.zeros(f.shape[0],num_m_bits).cuda()
    m = torch.cat([m1, m2, m3], dim=-1)

    dtype = f.type()
    idx = torch.arange(num_m_bits).unsqueeze(0).type(dtype) \
          + (8. - e_scientific).unsqueeze(-1)
    idx = idx.long()
    m = torch.gather(m, dim=-1, index=idx)

    return torch.cat([s, e, m], dim=-1).type(dtype)


def integer2bit(integer, num_bits=8):
  """Turn integer tensor to binary representation.
      Args:
          integer : torch.Tensor, tensor with integers
          num_bits : Number of bits to specify the precision. Default: 8.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = integer.type()
  exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
  exponent_bits = exponent_bits.repeat(integer.shape + (1,))
  out = integer.unsqueeze(-1) / 2 ** exponent_bits
  return (out - (out % 1)) % 2


def remainder2bit(remainder, num_bits=127):
  """Turn a tensor with remainders (floats < 1) to mantissa bits.
      Args:
          remainder : torch.Tensor, tensor with remainders
          num_bits : Number of bits to specify the precision. Default: 127.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = remainder.type()
  exponent_bits = torch.arange(num_bits).type(dtype)
  exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
  out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
  return torch.floor(2 * out)


if __name__ == "__main__":
    floats = np.random.random(10).astype(np.float32)*2-1
    bins = Array2Bin(floats)
    new_floats = Bin2Array(bins)

    same = floats == new_floats

    g=0


