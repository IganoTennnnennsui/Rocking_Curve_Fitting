import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz

# Voigt関数の定義
def voigt(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z))

# 曲線の最大値を求める関数
def find_peak(x, y):
    max_index = np.argmax(y)
    return x[max_index], y[max_index]

# FWHMを計算する関数
def calculate_fwhm(x, y, z):
    max_x, max_y = find_peak(x, y)
    half_max = max_y / 2.0

    # Half Maximumの値を計算
    half_max_value = max_y / 2.0

    # Half Maximumの値を更新
    half_max_value += z / 2.0

    # フィッティング曲線と半高の水平線の交点を求める
    idx_lower = np.argmax(y >= half_max_value)
    idx_upper = np.argmax(y[::-1] >= half_max_value)
    idx_upper = len(y) - idx_upper - 1

    # 交点のx座標を取得
    x1 = x[idx_lower]
    x2 = x[idx_upper]

    # FWHMを計算
    fwhm = x2 - x1

    return fwhm, half_max_value

# テキストファイルからデータの読み込み
def load_data(file_path):
    data = np.loadtxt(file_path)
    x_data = data[:, 0]
    y_data = data[:, 1]
    return x_data, y_data

# フィッティングに使用するデータファイルのパス
file_path = 'yourfile.txt'

# データの読み込み
x_data, y_data = load_data(file_path)

# 最小値から最大値まで0.0001ずつ増加する配列を生成
x_data_extended = np.arange(min(x_data), max(x_data) + 0.0001, 0.0001)

# 最初の10点のyの平均をバックグラウンドとして求める
background = np.mean(y_data[:10])

# 初期値の設定
initial_guess = [max(y_data), np.mean(x_data), 1.0, 0.5]

# Voigt関数のフィッティング
params, covariance = curve_fit(voigt, x_data, y_data, p0=initial_guess, maxfev=5000)

# FWHMおよびHalf Maximumの計算
fwhm, half_max_value = calculate_fwhm(x_data_extended, voigt(x_data_extended, *params), background)

# プロット
plt.plot(x_data, y_data, 'bo', label='Data')
plt.plot(x_data_extended, voigt(x_data_extended, *params), 'r-', label='Fit')
plt.axhline(y=half_max_value, color='gray', linestyle='--', label='Half Maximum')
plt.axvline(x=params[1] - fwhm/2, color='green', linestyle='--', label='FWHM')
plt.axvline(x=params[1] + fwhm/2, color='green', linestyle='--')

# バックグラウンドを表示
plt.plot(x_data, np.full_like(x_data, background), 'k--', label='Background')

# FWHMをグラフ上に表示
plt.annotate(f'FWHM: {fwhm:.4f}', xy=(params[1], max(y_data)/2), xytext=(params[1] + 0.2, max(y_data)/2),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.legend()
plt.show()

# FWHMおよびHalf Maximumの表示
print("FWHM:", fwhm)
print("Half Maximum:", half_max_value)
