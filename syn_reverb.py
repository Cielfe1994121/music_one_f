from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal  # 畳み込み用


class syn_reverb:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )

        # モノラル対策
        if self.data.ndim == 1:
            self.data = np.vstack([self.data, self.data])
        elif np.mean(np.abs(self.data[1])) < 0.0001:
            print("Mono detected, copying...")
            self.data[1] = self.data[0].copy()

        length = self.data.shape[1]

        # -----------------------------------------------------
        # 1. 人工リバーブ（Impulse Response）を作る
        # -----------------------------------------------------
        # 「シャー」というホワイトノイズを作って、フェードアウトさせる
        reverb_time = 2.0  # 残響の長さ（秒）。長いと洞窟、短いと部屋。
        ir_length = int(self.sr * reverb_time)

        # ノイズ生成
        impulse_response = np.random.randn(ir_length)
        # 減衰カーブ（Exponential Decay）を作る
        decay = np.exp(
            -np.linspace(0, 5, ir_length)
        )  # 5という数字を変えると減衰感が変わる
        impulse_response = impulse_response * decay

        # -----------------------------------------------------
        # 2. 畳み込み（Convolution）でWet音（残響のみの音）を作る
        # -----------------------------------------------------
        print("Convolving reverb... (Please wait)")
        wet_signal = np.zeros_like(self.data)

        # fftconvolveは高速な畳み込み計算
        # mode='full'だと長くなるので、元の長さと同じ部分だけ切り出す
        # ※ステレオなので左右別々に計算
        wet_left = signal.fftconvolve(self.data[0], impulse_response, mode="full")
        wet_right = signal.fftconvolve(self.data[1], impulse_response, mode="full")

        # 長さが伸びてしまうので、元の長さにカットする
        wet_signal[0] = wet_left[:length]
        wet_signal[1] = wet_right[:length]

        # 音量が爆発しがちなので、Dryと同じくらいに正規化（ノーマライズ）しておく
        wet_signal = wet_signal / np.max(np.abs(wet_signal)) * np.max(np.abs(self.data))

        # -----------------------------------------------------
        # 3. 1/fゆらぎで混ぜる（Mix）
        # -----------------------------------------------------
        self.one_f = ofg.generate_one_f(length)

        # 混ぜ具合（0.0 = 原音のみ 〜 0.5 = どっぷり残響）
        # ※ 1.0にすると原音が消えて「幽霊の声」になります
        mix_ratio = (self.one_f.ifft_real_result - 1) * 0.3  # 倍率調整
        # マイナスにならないように、かつ0.8を超えないように制限
        mix_ratio = np.clip(mix_ratio + 0.2, 0.0, 0.6)

        # Dry * (1 - ratio) + Wet * ratio
        self.data_mix = self.data * (1 - mix_ratio) + wet_signal * mix_ratio

        # 再生
        self.file.play_from_array(self.data_mix.T, self.sr)

    def get_beaf(self):
        return self.data, self.data_mix

    def vid(self, be, af):
        # グラフ描画（左チャンネルのみ）
        self.limit = int(1 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(be[0, : self.limit], label="Before (Dry)", color="tab:blue")
        ax[1].plot(af[0, : self.limit], label="After (Reverb Mix)", color="tab:purple")

        # 重ねて比較
        ax[2].plot(be[0, : self.limit], label="Before", color="tab:blue", alpha=0.6)
        ax[2].plot(af[0, : self.limit], label="After", color="tab:purple", alpha=0.6)

        for a in ax:
            # ズームは自動（コメントアウト推奨）
            # a.set_ylim(-1.0, 1.0)
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_reverb()
    syn.get_file_path()
    b, a = syn.get_beaf()
    syn.vid(b, a)
