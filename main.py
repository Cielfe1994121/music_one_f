from gui_play import gui_play as gp
import one_f_generator as ofg
import syn_pan as pan
import syn_pitch as pit
import syn_volume as vol
import syn_timbre as tim
import syn_reverb as rev

if __name__ == "__main__":
    get_file = gp()
    file = get_file.gui_get_music()
