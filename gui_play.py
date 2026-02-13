import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import soundfile as sf
import numpy as np
import os


class gui_play:
    def __init__(self):
        # サンプリングレートの初期値（読み込み時に更新される）
        self.sr = 44100

    def gui_get_music(self) -> str:
        """
        GUIで音楽ファイルを選択させる
        :return: 選択されたファイルのパス（選択キャンセルの場合は空文字）
        """
        # ルートウィンドウを作成し、即座に隠す
        root = tk.Tk()
        root.withdraw()

        # ファイル選択ダイアログ
        # ユーザーが変なファイルを選ばないようにフィルタをかける
        file_path = filedialog.askopenfilename(
            title="音楽ファイルを選択してください",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.flac *.m4a"),
                ("All Files", "*.*"),
            ],
        )

        # 用が済んだらウィンドウリソースを解放
        root.destroy()

        if not file_path:
            print("No file selected.")
            return ""

        return file_path

    def play_music(self, file_path: str):
        """
        ファイルパスから直接再生する（プレビュー用）
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found -> {file_path}")
            return

        try:
            print(f"Playing: {os.path.basename(file_path)} ...")
            # always_2d=True で常に(サンプル数, チャンネル数)の形にする
            data, self.sr = sf.read(file_path, always_2d=True)

            sd.play(data, self.sr)
            sd.wait()  # 再生終了まで待機
            print("Playback finished.")

        except Exception as e:
            print(f"Playback Error: {e}")

    def play_from_array(self, data: np.ndarray, sr: int):
        """
        NumPy配列から音声を再生する
        """
        try:
            # sounddeviceは float32 推奨
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # 転置チェック: (channels, samples) なら (samples, channels) に直す
            # sounddeviceは (行=時間, 列=チャンネル) を期待する
            if data.ndim == 2 and data.shape[0] < data.shape[1]:
                # 明らかにチャンネル数 < サンプル数 の場合のみ転置
                # (2, 200000) -> (200000, 2)
                data = data.T

            print("Starting playback from array...")
            sd.play(data, sr)
            sd.wait()  # 最後まで聴く場合
            print("Playback finished.")

        except Exception as e:
            print(f"Array Playback Error: {e}")


# テスト用
if __name__ == "__main__":
    player = gui_play()
    path = player.gui_get_music()
    if path:
        player.play_music(path)
