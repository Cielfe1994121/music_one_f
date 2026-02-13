from gui_play import gui_play as gp
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# インポート（ここは現状のファイル名とクラス名に合わせています）
from syn_volume import syn_volume
from syn_pan import syn_pan
from syn_pitch import syn_pitch
from syn_timbre import syn_timbre
from syn_reverb import syn_reverb

if __name__ == "__main__":
    player = gp()
    file_path = player.gui_get_music()
    if not file_path:
        exit()

    print("Loading music...")
    data, sr = librosa.load(file_path, mono=False, sr=None, duration=180)

    be_data = data.copy()

    # 1. Volume (メソッド名が syn_vol のはず)
    vol_inst = syn_volume()
    data = vol_inst.syn_vol(data, sr)
    data = data / np.max(np.abs(data))
    print("volumeを1/fに乗せてる")

    # 2. Pan (メソッド名が syn_pan のはず)
    pan_inst = syn_pan()
    data = pan_inst.syn_pan(data, sr)
    data = data / np.max(np.abs(data))
    print("panを1/fに乗せてる")

    # 3. Pitch
    # 【修正箇所】エラー通り、メソッド名を syn_pit に合わせる！
    pit_inst = syn_pitch()
    data = pit_inst.syn_pit(data, sr)  # syn_pitch ではなく syn_pit
    data = data / np.max(np.abs(data))
    print("pitchを1/fに乗せてる")

    # 4. Timbre
    # 【修正箇所】ここもおそらく syn_tim になっているはず！
    tim_inst = syn_timbre()
    data = tim_inst.syn_tim(data, sr)  # syn_timbre ではなく syn_tim
    data = data / np.max(np.abs(data))
    print("timbreを1/fに乗せてる")

    # 5. Reverb
    # 【修正箇所】ここもおそらく syn_rev になっているはず！
    rev_inst = syn_reverb()
    data = rev_inst.syn_rev(data, sr)  # syn_reverb ではなく syn_rev
    data = data / np.max(np.abs(data))
    print("reverbを1/fに乗せてる")

    print("Playing...")
    player.play_from_array(data.T, sr)
    """
    af = data
    # 1秒分だけ表示
    limit = int(1 * sr)
    fig, ax = plt.subplots(3, 1, sharex=True)

    # ステレオ対応：左チャンネル[0]だけをプロット
    ax[0].plot(be[0, :limit], label="Before", color="tab:blue")
    ax[1].plot(af[0, :limit], label="After", color="tab:orange")

    ax[2].plot(be[0, :limit], label="Before", color="tab:blue", alpha=1)
    ax[2].plot(af[0, :limit], label="After", color="tab:orange", alpha=0.5)

    for a in ax:
        # ズームは自動
        # a.set_ylim(-0.01, 0.01)
        a.legend(loc="upper right")
        a.grid(True, linestyle="--", alpha=0.5)

    plt.show()
    """

    # --- 1. 波形の「極小ズーム」でギザギザ（音割れ）を見る ---
    # 0.01秒分（約441サンプル）だけ切り出す
    zoom_limit = int(0.01 * sr)
    start_sample = int(0.5 * sr)  # 曲の開始0.5秒地点から

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Before/Afterを重ねて表示
    ax[0].plot(
        be_data[0, start_sample : start_sample + zoom_limit],
        label="Original (Smooth)",
        alpha=0.8,
    )
    ax[0].plot(
        data[0, start_sample : start_sample + zoom_limit],
        label="Processed (Distorted)",
        alpha=0.6,
    )
    ax[0].set_title("Waveform Zoom (0.01s) - Look for jagged edges")
    ax[0].legend()

    # --- 2. スペクトル表示で「ノイズ」を暴く ---
    # 周波数成分を計算
    f_be, Pxx_be = signal.welch(be_data[0], sr, nperseg=1024)
    f_af, Pxx_af = signal.welch(data[0], sr, nperseg=1024)

    ax[1].semilogy(f_be, Pxx_be, label="Before")
    ax[1].semilogy(f_af, Pxx_af, label="After (1/f Processed)")
    ax[1].set_title("Power Spectral Density - 1/f slope & Artifacts")
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Power")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
