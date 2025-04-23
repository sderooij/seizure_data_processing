from seizure_data_processing import EEG
from seizure_data_processing.config import TUSZ_DIR, MIT_CHB_DIR

# class TestEEG:
#
#
# 	def test_load(self):
#
# 		return


def test_eeg_init():

    file = (
        TUSZ_DIR
        + r"edf\train\01_tcp_ar\000\00000077\s003_2010_01_21\00000077_s003_t000.edf"
    )
    channels = ["FP1", "FP2", "F3", "F4", "C3", "C4", "CZ"]
    eeg = EEG(file, channels)
    assert eeg._dataset == "tusz"
    assert len(eeg.data) == len(channels)
    assert eeg.Fs == 250
    assert eeg.annotations.empty


def test_get_labels():

    file = MIT_CHB_DIR + "chb01/chb01_03.edf"
    channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3"]
    eeg = EEG(file, channels)
    time_vec = eeg.get_time()
    labels = eeg.get_labels()
    assert labels[0] == -1
    assert labels[time_vec == 3000.0] == 1
