from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt
import numpy as np


class syn_pan:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        self.data, self.sr = librosa.load(
            self.file_path, mono=False, sr=None, duration=180
        )
        return self.data, self.sr

    def syn_pan(self, data, sr):
        # 【修正1】データが1次元（モノラル）だった場合の強制ステレオ化
        # これをやらないと次の行の data[1] でエラーになります
        if data.ndim == 1:
            data = np.vstack([data, data])
        # ステレオだけど片方が無音の場合のコピー処理
        elif np.mean(np.abs(data[1])) < 0.0001:
            print("Mono source detected! Copying Left to Right...")
            data[1] = data[0].copy()

        # 【修正2】self.data ではなく、引数の data の長さを見る
        # (Mainで他の加工済みデータが渡されても動くようにするため)
        length = data.shape[1]
        self.one_f = ofg.generate_one_f(length)

        # 計算式はそのままでOK（巧みな反転ロジックですね！）
        # 左：ゆらぎを引く（1 - result がマイナス成分になるのを利用）
        # 右：ゆらぎを足す
        data[0] = data[0] * (1 + (1 - self.one_f.ifft_real_result))
        data[1] = data[1] * (1 - (1 - self.one_f.ifft_real_result))

        # 【修正3】加工したデータを返す（バケツリレー用）
        return data

    def get_lfri(self):
        # 加工後のデータが self.data に反映されている前提
        return self.data[0], self.data[1]

    def vid(self, lf, ri):
        self.limit = int(1 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(lf[: self.limit], label="Left", color="tab:blue")
        ax[1].plot(ri[: self.limit], label="Right", color="tab:orange")

        ax[2].plot(lf[: self.limit], label="Left", color="tab:blue", alpha=1)
        ax[2].plot(ri[: self.limit], label="Right", color="tab:orange", alpha=0.5)

        for a in ax:
            # 拡大表示するために範囲固定を解除
            # a.set_ylim(-0.01, 0.01)
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_pan()
    data, sr = syn.get_file_path()

    # 戻り値を受け取る形にする
    data = syn.syn_pan(data, sr)

    syn_lf, syn_ri = syn.get_lfri()
    syn.vid(syn_lf, syn_ri)
