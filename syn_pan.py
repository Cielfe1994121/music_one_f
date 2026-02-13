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
        if np.mean(np.abs(self.data[1])) < 0.0001:
            print("Mono source detected! Copying Left to Right...")
            self.data[1] = self.data[0].copy()

        # print(self.data)
        self.one_f = ofg.generate_one_f(self.data.shape[1])
        self.data[0] = self.data[0] * (1 + (1 - self.one_f.ifft_real_result))
        self.data[1] = self.data[1] * (1 - (1 - self.one_f.ifft_real_result))
        self.file.play_from_array(self.data.T, self.sr)

    def get_lfri(self):
        self.le = self.data[0]
        self.ri = self.data[1]
        return self.data[0], self.data[1]

    def vid(self, lf, ri):
        self.limit = int(1 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(lf[: self.limit], label="Left", color="tab:blue")
        ax[1].plot(ri[: self.limit], label="Right", color="tab:orange")

        ax[2].plot(lf[: self.limit], label="Left", color="tab:blue", alpha=1)
        ax[2].plot(ri[: self.limit], label="Right", color="tab:orange", alpha=0.5)

        for a in ax:
            a.set_ylim(-0.01, 0.01)
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_pan()
    syn_play = syn.get_file_path()
    syn_lf, syn_ri = syn.get_lfri()
    syn_vid = syn.vid(syn_lf, syn_ri)
