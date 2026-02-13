from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal  # フィルター用


class syn_timbre:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )

        # モノラル対策（Panと同じ）
        if self.data.ndim == 1:
            self.data = np.vstack([self.data, self.data])
        elif np.mean(np.abs(self.data[1])) < 0.0001:
            self.data[1] = self.data[0].copy()

        length = self.data.shape[1]

        # 1. 「こもった音（Muffled）」を作る
        # バターワースフィルタ（ローパス）を作成
        # カットオフ周波数：1000Hz（これより高い音を削る＝こもる）
        nyquist = self.sr / 2
        cutoff = 1000 / nyquist
        b, a = signal.butter(4, cutoff, btype="low")  # 4次は急峻さ

        # フィルターを適用（lfilterは高速！）
        muffled_data = np.zeros_like(self.data)
        muffled_data[0] = signal.lfilter(b, a, self.data[0])
        muffled_data[1] = signal.lfilter(b, a, self.data[1])

        # 2. 1/fゆらぎ係数を作る
        self.one_f = ofg.generate_one_f(length)
        # 0.0 〜 1.0 の範囲に正規化する（ブレンド率にするため）
        # ※ここでの調整がキモです
        mix_ratio = (self.one_f.ifft_real_result - 1) * 5.0  # 倍率で揺れ幅調整
        # 0〜1の範囲からはみ出ないようにクリップする
        mix_ratio = np.clip(mix_ratio + 0.5, 0.0, 1.0)

        # 3. ブレンド（クロスフェード）
        # Original * (1 - ratio) + Muffled * ratio
        self.data_mix = self.data * (1 - mix_ratio) + muffled_data * mix_ratio

        # 再生
        # self.file.play_from_array(self.data_mix.T, self.sr)

    def get_beaf(self):
        self.be = self.data
        self.af = self.data_mix
        return self.be, self.af

    def vid(self, be, af):
        self.limit = int(1 * self.sr)  # 1秒分
        fig, ax = plt.subplots(3, 1, sharex=True)

        # 【修正】be[0, :self.limit] にする（左チャンネルだけ見る）
        # ステレオデータの [行, 列] を指定する書き方です
        ax[0].plot(be[0, : self.limit], label="Before", color="tab:blue")
        ax[1].plot(af[0, : self.limit], label="After (Muffled Mix)", color="tab:orange")

        ax[2].plot(be[0, : self.limit], label="Before", color="tab:blue", alpha=1)
        ax[2].plot(af[0, : self.limit], label="After", color="tab:orange", alpha=0.5)

        for a in ax:
            # 音量は -1.0 〜 1.0 なので固定
            # a.set_ylim(-1.0, 1.0)
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_timbre()
    syn_play = syn.get_file_path()
    syn_be, syn_af = syn.get_beaf()
    syn_vid = syn.vid(syn_be, syn_af)
