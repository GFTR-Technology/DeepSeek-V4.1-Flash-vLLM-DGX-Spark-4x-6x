# GB10 GPU clock latch: two of four Sparks were stuck below 1 GHz (2026-09-10)

## Symptom

Two of the four Sparks, Reddie (NVIDIA FE) and Asusi (ASUS GX10), ran their GPU at 631-949 MHz. That held at idle and under the eager-mode load of boot 6. The other two, Spark4 and Bluey, ran at 2177-2561 MHz.

On the stuck pair, `nvidia-smi` showed nothing abnormal:
- P0, persistence on
- application clocks 2418 MHz, max 3003 MHz
- no active clock event reasons (no power cap, thermal, or HW slowdown), no locked clocks
- clean kernel logs

In a TP4 deployment every collective waits for the slowest rank, so two slow GPUs set the pace for all four.

## Cause

The embedded controller (EC) makes the GPU's DVFS decisions below the OS, and it can latch into a low-clock state. This matches the write-up in [ShiningMeUp/dgx-spark-gpu-underclock-fix](https://github.com/ShiningMeUp/dgx-spark-gpu-underclock-fix) and an NVIDIA forum report of a clock pinned under load that ignored `nvidia-smi -lgc`.

A normal reboot or soft shutdown does not clear the latch. The EC keeps standby power while the adapter is plugged in.

## Fix

1. Shut down.
2. **Unplug the power adapter for 30-60 s or longer.**
3. Plug it back in and power on.

While the adapter is out, check that it is the original and fully seated; the EC also limits clocks when it sees insufficient power.

## Check

15 s fp16 4096x4096 matmul burn, with `nvidia-smi` sampled at 12 s (`tools/recover.sh`, the burn part):

| node | before | after power-cycle |
|---|---|---|
| Reddie | idle 890 MHz, 5.3 W | **84.0 TFLOPS**, 2294 MHz, 94.6 W under load; idles at 208 MHz |
| Asusi | idle 637 MHz, 4.6 W | **76.6 TFLOPS**, 2333 MHz, 90.6 W under load; idles at 208 MHz |
| Spark4 (not latched) | idle 2411 MHz | 89.2 TFLOPS, 2301 MHz, 93.4 W |
| Bluey (not latched) | idle 2411 MHz | 88.8 TFLOPS, 2177 MHz, 92.9 W |

**Rule of thumb:**
- Under load, a healthy GB10 draws 80 W or more at about 2.2-2.4 GHz and does 75-90 TFLOPS fp16.
- A latched one sits near 700-950 MHz and under 20 W.

**Run the burn check before trusting any benchmark on a Spark fleet.** The latch shows no symptoms except speed.
