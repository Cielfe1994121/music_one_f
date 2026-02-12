import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import soundfile as sf


class gui_play:
    def gui_get_music(self):
        r = tk.Tk()
        r.withdraw()
        self.file = filedialog.askopenfilename(title=".mp3を選択してください")
        return self.file

    def play_music(self, file_path):
        self.sig, self.sr = sf.read(file_path, always_2d=True)
        sd.play(self.sig, self.sr)
        print("start")
        sd.play(self.sig, self.sr)
        sd.wait()
        print("end")

    def play_from_array(self, data, sr):
        sd.play(data, sr)
        sd.wait()


if __name__ == "__main__":  # テスト用コード
    music = gui_play()
    music_file = music.gui_get_music()
    music.play_music(music_file)
