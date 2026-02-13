from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np


class syn_pitch:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        # Pitch処理でもステレオ対応は必須
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )

        # モノラル対策（Panと同じ）
        if self.data.ndim == 1:  # 1次元配列なら
            self.data = np.vstack([self.data, self.data])  # 無理やり2次元にする
        elif np.mean(np.abs(self.data[1])) < 0.0001:
            print("Mono source detected! Copying Left to Right...")
            self.data[1] = self.data[0].copy()

        # ---------------------------------------------------------
        # 【ここから Pitch処理（Tape Wobble）】
        # ---------------------------------------------------------

        length = self.data.shape[1]
        self.one_f = ofg.generate_one_f(length)

        # 1. ゆらぎ係数を作る
        # ピッチは敏感なので、係数は小さめにしないと「酔い」ます
        # * 0.02 くらいで「お、テープ伸びてるな」と分かります
        fluctuation = (self.one_f.ifft_real_result - 1) * 0.05

        # 2. 「歪んだ時間の地図」を作る
        # 1.0 = 普通の速度。 1.05 = 速い。 0.95 = 遅い。
        speed_map = 1.0 + fluctuation

        # 「累積和（cumsum）」を使って、歪んだ時間を積み上げる
        # 例：[1, 1, 1] -> [1, 2, 3] (正常)
        # 例：[1, 1.2, 0.8] -> [1, 2.2, 3.0] (歪んでる)
        dirty_time_index = np.cumsum(speed_map)

        # 3. 尺合わせ（重要！）
        # 早回し・遅回しをすると、曲の長さが変わってしまいます。
        # 強制的に「元の曲の長さ」に収まるように縮尺を合わせます。
        dirty_time_index = dirty_time_index / dirty_time_index[-1] * (length - 1)

        # 4. リサンプリング（補間）
        # 「歪んだ時間の地図」に従って、元のデータから音を拾ってくる
        # np.interp(欲しい場所, 今の目盛り, 今のデータ)
        original_index = np.arange(length)

        # 左チャンネル処理
        self.data[0] = np.interp(dirty_time_index, original_index, self.data[0])
        # 右チャンネル処理（同じ歪み方をさせないと左右でズレて気持ち悪いので同じindexを使う）
        self.data[1] = np.interp(dirty_time_index, original_index, self.data[1])

        # 再生（転置を忘れずに）
        self.file.play_from_array(self.data.T, self.sr)

    def get_beaf(self):
        # Pitchの変化は見えにくいので、今回は「時間の歪みカーブ」そのものを返してもいいかも
        # とりあえず波形を返します
        return self.data[0], self.data[1]  # 加工後のLR

    def vid(self, left, right):
        # Pitchの可視化は難しいので、とりあえずPanと同じ構成で確認
        self.limit = int(10 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(left[: self.limit], label="Left (Pitch Mod)", color="tab:blue")
        ax[1].plot(right[: self.limit], label="Right (Pitch Mod)", color="tab:orange")

        # 拡大比較
        ax[2].plot(left[: self.limit], label="Left", color="tab:blue", alpha=0.7)
        ax[2].plot(right[: self.limit], label="Right", color="tab:orange", alpha=0.7)

        for a in ax:
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_pitch()
    syn.get_file_path()
    l, r = syn.get_beaf()
    syn.vid(l, r)
