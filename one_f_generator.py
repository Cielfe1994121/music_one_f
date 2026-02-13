import numpy as np
import matplotlib.pyplot as plt


class generate_one_f:
    def __init__(self, duration: int):
        """
        初期化と同時に1/fゆらぎを生成します。
        :param duration: 生成するサンプルの長さ
        """
        self.duration = duration
        self.ifft_real_result = self._generate_pink_noise(duration)

    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """
        1/fゆらぎ（ピンクノイズ）を生成する内部メソッド
        """
        # 1. ホワイトノイズ生成 (-0.5 〜 0.5)
        white_noise = np.random.rand(n_samples) - 0.5

        # 2. FFT（周波数領域へ変換）
        fft_result = np.fft.fft(white_noise)

        # 3. 1/f特性の適用 (振幅を 1/sqrt(f) でスケーリング)
        # 0番目(DC成分)の除算エラーを防ぐため 1 を代入
        freqs = np.arange(n_samples)
        freqs[0] = 1

        # ピンクノイズ化フィルタ
        pink_spectrum = fft_result / np.sqrt(freqs)

        # 4. IFFT（時間領域へ戻す）
        ifft_result = np.fft.ifft(pink_spectrum)

        # 5. 実部の取り出しとスケーリング
        # ※ 既存のチューニングを維持するため、元の係数(* 100 + 1)を保持
        final_signal = ifft_result.real * 100 + 1

        return final_signal

    def one_f_visualize(self):
        """生成されたゆらぎをグラフで確認"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.ifft_real_result)
        plt.title("1/f Fluctuation (Pink Noise)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()


# テスト用
if __name__ == "__main__":
    gen = generate_one_f(1000)
    print(f"Mean: {np.mean(gen.ifft_real_result):.4f}")
    gen.one_f_visualize()
