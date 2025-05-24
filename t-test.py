from scipy.stats import ttest_ind

# 95th percentile response times (ms)
lift_shift_rts = [1200, 950, 800, 1100, 1340]
cloud_native_rts = [200, 220, 240, 250, 300, 330, 400, 460, 500, 560]

t_stat, p = ttest_ind(lift_shift_rts, cloud_native_rts, equal_var=False)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p:.5f}")
