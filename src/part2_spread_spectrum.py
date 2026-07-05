"""Part 2: direct-sequence spread spectrum experiment."""

import numpy as np

from utils import (
    add_awgn,
    add_narrowband_interference,
    bpsk_demodulate,
    bpsk_modulate,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
    plot_correlation_snapshot,
)


def _validate_pn_chips(pn_chips):
    pn_chips = np.asarray(pn_chips, dtype=float)
    if pn_chips.ndim != 1 or len(pn_chips) == 0:
        raise ValueError('pn_chips must be a non-empty one-dimensional array')
    if not np.all(np.isin(pn_chips, [-1, 1])):
        raise ValueError('pn_chips must contain only +1 and -1')
    return pn_chips


def generate_m_sequence(register_state, taps, length=None):
    """
    Generate a bipolar m-sequence with an LFSR.

    Convention:
        register_state is listed from left to right.
        taps are 1-based positions from left to right.
        each clock outputs the rightmost bit, shifts right, and inserts feedback
        at the left. The feedback bit is XOR of tapped bits.

    Returns:
        chips in bipolar form: bit 0 -> +1, bit 1 -> -1.
    """
    state = np.asarray(register_state, dtype=int)
    taps = list(taps)
    if state.ndim != 1 or len(state) == 0:
        raise ValueError('register_state must be a non-empty one-dimensional array')
    if not np.all((state == 0) | (state == 1)) or not np.any(state):
        raise ValueError('register_state must be binary and not all zeros')
    if not taps or any(tap < 1 or tap > len(state) for tap in taps):
        raise ValueError('taps must be valid 1-based register positions')
    if length is None:
        length = 2 ** len(state) - 1
    if length <= 0:
        raise ValueError('length must be positive')

    # LFSR: 每时钟输出最右位，右移，反馈在左侧插入
    chips = np.zeros(length, dtype=float)
    state = state.copy()
    for i in range(length):
        # 输出最右位: 0→+1, 1→-1
        output_bit = state[-1]
        chips[i] = 1.0 if output_bit == 0 else -1.0
        # 反馈位 = XOR of tapped bits (1-based taps)
        feedback = 0
        for tap in taps:
            feedback ^= state[tap - 1]
        # 右移，反馈插入最左
        state = np.roll(state, 1)
        state[0] = feedback
    return chips


def dsss_spread(bits, pn_chips):
    """
    Spread BPSK symbols with PN chips.

    For each bit, map 0 -> +1 and 1 -> -1, then multiply by the whole PN
    sequence. Output length is len(bits) * len(pn_chips).
    """
    bits = np.asarray(bits, dtype=int)
    pn_chips = _validate_pn_chips(pn_chips)
    if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits must be a one-dimensional binary array')

    # BPSK映射: 0→+1, 1→-1, 再与PN码逐片相乘
    bpsk = 1.0 - 2.0 * bits.astype(float)  # shape: (num_bits,)
    spreading_factor = len(pn_chips)
    chips = np.zeros(len(bits) * spreading_factor, dtype=float)
    for i in range(len(bits)):
        start = i * spreading_factor
        chips[start:start + spreading_factor] = bpsk[i] * pn_chips
    return chips


def dsss_despread(received_chips, pn_chips):
    """
    Despread received chips by correlation with the same PN sequence.

    Returns:
        recovered bits after hard decision. Non-negative correlation -> bit 0.
    """
    received_chips = np.asarray(received_chips, dtype=float)
    pn_chips = _validate_pn_chips(pn_chips)
    if received_chips.ndim != 1 or len(received_chips) % len(pn_chips) != 0:
        raise ValueError('received_chips length must be a multiple of PN length')

    # 按扩频因子reshape，与PN码相关，非负相关→bit 0
    spreading_factor = len(pn_chips)
    num_bits = len(received_chips) // spreading_factor
    matrix = received_chips.reshape(num_bits, spreading_factor)
    correlations = matrix @ pn_chips / spreading_factor
    return (correlations < 0).astype(int)


def processing_gain_db(spreading_factor):
    """Return processing gain 10*log10(spreading_factor) in dB."""
    if spreading_factor <= 0:
        raise ValueError('spreading_factor must be positive')

    return 10.0 * np.log10(spreading_factor)


def despread_with_timing_offset(received_chips, pn_chips, max_offset):
    """Optional: search timing offset by maximum correlation magnitude."""
    if max_offset < 0:
        raise ValueError('max_offset must be non-negative')

    # 在偏移范围内搜索最佳同步位置（最大相关幅度）
    spreading_factor = len(pn_chips)
    num_bits = len(received_chips) // spreading_factor
    best_offset = 0
    best_total = -np.inf
    for offset in range(-max_offset, max_offset + 1):
        shifted = np.roll(received_chips, offset)
        shifted[:max(0, offset)] = 0
        shifted[len(shifted) + min(0, offset):] = 0
        matrix = shifted.reshape(num_bits, spreading_factor)
        correlations = np.abs(matrix @ pn_chips / spreading_factor)
        total = np.sum(correlations)
        if total > best_total:
            best_total = total
            best_offset = offset
    # 用最佳偏移解扩
    aligned = np.roll(received_chips, best_offset)
    return dsss_despread(aligned, pn_chips)


def _correlation_values(received_chips, pn_chips):
    matrix = np.asarray(received_chips, dtype=float).reshape(-1, len(pn_chips))
    return matrix @ np.asarray(pn_chips, dtype=float) / len(pn_chips)


def run_spread_spectrum_demo():
    """Run Part 2 demo and generate figures."""
    print('=' * 60)
    print('Part 2: DSSS 扩频通信实验')
    print('=' * 60)
    snr_db_values = np.array([-6, -3, 0, 3, 6, 9], dtype=float)

    try:
        pn_chips = generate_m_sequence([1, 1, 1, 0, 1], taps=[5, 2], length=31)
        bits = generate_bits(3000, seed=2026)
        unspread_ber = []
        dsss_ber = []

        for index, snr_db in enumerate(snr_db_values):
            symbols = bpsk_modulate(bits)
            unspread_rx = add_narrowband_interference(symbols, amplitude=0.8, frequency=0.11)
            unspread_rx = add_awgn(unspread_rx, snr_db, seed=100 + index)
            unspread_ber.append(calculate_ber(bits, bpsk_demodulate(unspread_rx)))

            chips = dsss_spread(bits, pn_chips)
            rx_chips = add_narrowband_interference(chips, amplitude=0.8, frequency=0.11)
            rx_chips = add_awgn(rx_chips, snr_db, seed=200 + index)
            recovered = dsss_despread(rx_chips, pn_chips)
            dsss_ber.append(calculate_ber(bits, recovered))

        plot_ber_curve(
            snr_db_values,
            {'未扩频': unspread_ber, f'DSSS(N={len(pn_chips)})': dsss_ber},
            '窄带干扰下 DSSS 扩频前后 BER 对比',
            'dsss_ber_curve.png',
        )

        demo_bits = generate_bits(120, seed=77)
        demo_chips = dsss_spread(demo_bits, pn_chips)
        demo_rx = add_narrowband_interference(demo_chips, amplitude=0.8, frequency=0.11)
        demo_rx = add_awgn(demo_rx, 0, seed=88)
        correlations = _correlation_values(demo_rx, pn_chips)
        plot_correlation_snapshot(correlations, 'dsss_correlation_snapshot.png')

        print(f'[OK] 处理增益: {processing_gain_db(len(pn_chips)):.2f} dB')
        print('[OK] 已生成 results/dsss_ber_curve.png')
        print('[OK] 已生成 results/dsss_correlation_snapshot.png')
    except NotImplementedError as error:
        print(f'[WAIT] 尚未完成核心函数: {error}')
    except Exception as error:
        print(f'[FAIL] Part 2 运行失败: {error}')


if __name__ == '__main__':
    run_spread_spectrum_demo()
