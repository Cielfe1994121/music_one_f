from gui_play import gui_play as gp
import one_f_generator as ofg
import librosa 

class synthesis_one_f:
    def get_file_path(self):
        file = gp()
        return file.gui_get_music()






if __name__ == "__main__":
    syn = synthesis_one_f()
    syn_file = syn.get_file_path()
    print(syn_file)