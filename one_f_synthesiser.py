from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa
import matplotlib.pyplot as plt


class syn_vol:
    def get_file_path(self):
        self.file = gp()
        self.file_path = self.file.gui_get_music()
        self.data, self.sr = librosa.load(self.file_path, sr=None, duration=180)
        print(self.data)
        self.one_f = ofg.generate_one_f(len(self.data))
        self.one_f_data = self.data * self.one_f.ifft_real_result
        # self.file.play_from_array(self.one_f_data, self.sr)

    def get_beaf(self):
        self.be = self.data
        self.af = self.one_f_data
        return self.be, self.af

    def vid(self, be, af):
        self.limit = int(1 * self.sr)
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(be[: self.limit], label="Before", color="tab:blue")
        ax[1].plot(af[: self.limit], label="After", color="tab:orange")

        ax[2].plot(be[: self.limit], label="Before", color="tab:blue", alpha=1)
        ax[2].plot(af[: self.limit], label="After", color="tab:orange", alpha=0.5)

        for a in ax:
            a.set_ylim(-0.01, 0.01)
            a.legend(loc="upper right")
            a.grid(True, linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    syn = syn_vol()
    syn_play = syn.get_file_path()
    syn_be, syn_af = syn.get_beaf()
    syn_vid = syn.vid(syn_be, syn_af)
