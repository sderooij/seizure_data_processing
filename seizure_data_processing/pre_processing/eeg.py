"""
Definition of EEG class object
"""

# external libraries
import numpy as np
import pyedflib
from scipy import signal
import mne
from scipy.io import savemat
import h5py
from pathlib import Path

# internal libraries
from seizure_data_processing.datasets import mit_chb as chb
from seizure_data_processing.datasets import tusz, seize_it
from seizure_data_processing.pre_processing import features as ff


class EEG:
    """EEG class object"""

    # ---------------- Constructor ----------------------------------
    def __init__(self, filename: str, channels=None, *, dataset=""):
        """Initialize EEG object by loading an eeg file. The dataset is automatically detected. For now only support for MIT-CHB and TUSZ dataset

        Args:
            filename (str): Path of EEG file
            channels (list[str], optional): List of channels to extract. If None all channels are extracted. Defaults to None.
            dataset (str, optional): Name of the EEG dataset. "mit-chb" or "tusz". Defaults to "".
        """
        self._orig_channels = None
        self._file_duration = None
        self.filename = filename
        self.channels = channels
        self.data = np.zeros(1)  # channels x time
        self.Fs = None  # array or int
        self.annotations = None  # dataframe with cols start, stop, seiz_type, prob

        # self.load()
        # self._num_channels = len(self.channels)
        # self._num_samples = len(self.data[0])

        if "chb" in filename:
            self._dataset = "mit-chb"
        elif "s0" in filename:  # change for something better ?
            self._dataset = "tusz"
        elif "P_ID" in filename:
            self._dataset = "seize-it"
        else:
            self._dataset = dataset
        # self.annotate()

        self.features = None

    # --------------------- MAGIC METHODS -----------------------------------
    def __str__(self):
        return str(self.__class__) + ":" + str(self.__dict__)

    def __repr__(self):
        return str(self.__class__) + ":" + str(self.__dict__)

    # --------------------- PROPERTIES -----------------------------------
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        if data is not None and hasattr(data[0], "__len__"):
            self._num_samples = len(data[0])
        else:
            self._num_samples = 0

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channels):
        self._channels = channels
        if channels is not None:
            self._num_channels = len(channels)
        else:
            self._num_channels = 0

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def num_samples(self):
        return self._num_samples

    # --------------------- CLASS METHODS -----------------------------------
    def load(self):
        """load EEG signals from an EDF file

        Args:
            self (str): EEG object

        Returns:
            EEG object
        """
        try:
            f = pyedflib.EdfReader(self.filename)
        except IOError:
            print("Failed to open %s" % self.filename)
            raise

        # ----------- get channels and fs --------------
        channels = f.getSignalLabels()
        self._orig_channels = channels
        num_channels = f.signals_in_file
        fs = f.getSampleFrequencies()  # sampling frequency

        if self.channels is None:
            if any(fs == 1):
                rm_indices = np.where(fs == 1)[0]
                fs = np.delete(fs, rm_indices)
                if np.all(fs == fs[0]):
                    fs = fs[0]
                channels = np.delete(channels, rm_indices)
            else:
                rm_indices = []
                if np.all(fs == fs[0]):
                    fs = fs[0]

            signals = []
            for i in range(num_channels):
                if i in rm_indices:
                    continue
                signals.append(f.readSignal(i))

            try:
                self.data = np.array(signals, dtype=object)
            except:
                self.data = signals

            try:
                self.Fs = float(fs)
            except:
                self.Fs = fs
            self.channels = channels
            f.close()
            return self

        # --------- channel selection -----------
        # get indices of the selected channels (/labels)
        ch_indices = get_pos_edf(channels, self.channels)

        fs = fs[ch_indices]  # fs of selected channels

        if np.all(fs == fs[0]):
            fs = fs[0]

        # ------------------ read signals --------------------
        signals = []
        for i in ch_indices:
            signals.append(f.readSignal(i))

        # try:
        #     signals = np.transpose(signals)  # cols different channels, rows time
        # except:
        #     raise Exception("Different length of signals")

        self.data = np.array(signals, dtype=object)
        self.Fs = float(fs)

        self._file_duration = f.getFileDuration()

        f.close()
        return self

    def resample(self, new_fs):
        """
        Resample edf data to new sampling frequency.

        Args:
            self
            new_fs (int): new (desired) sampling frequency

        Returns:
           ndarray: array with resampled data

        """
        if self.data is None:
            self.load()

        if np.all(self.Fs == new_fs):
            return self
        elif not np.isscalar(self.Fs):
            raise Exception(
                "Original sampling frequency not a scalar, this method is not defined for such cases."
            )

        # number of points new signal
        num_sec = int(self.data.shape[1] / self.Fs)
        self.data = self.data[
            :, 0 : int(num_sec * self.Fs)
        ]  # to ensure correct resampling
        assert (self.data.shape[1] % self.Fs) == 0

        num_points = num_sec * new_fs
        self.data = signal.resample(
            self.data, num_points, axis=1, window=None, domain="time"
        )
        return self

    def annotate(self):
        """Annotate the EEG data

        Raises:
            Exception: If dataset other than mit-chb or tusz is supplied

        Returns:
            EEG: annotated EEG object
        """

        if self._dataset == "mit-chb":
            self.annotations = chb.load_annotations(self.filename)
        elif self._dataset == "tusz":
            self.annotations = tusz.load_annotations(self.filename)
        elif self._dataset == "seize-it":
            self.annotations = seize_it.load_annotations(self.filename)
        else:
            raise Exception("Dataset not supported for annotation.")

        return self

    def apply_montage(self, montage):
        """apply a montage to the EEG signals

        Args:
            self (EEG): EEG object
            montage (list[str]): list of montage

        Returns:
            self
        """

        # Split montage
        montage_split = np.array([part.split("-") for part in montage])
        # first + second index
        indices_first = get_pos_edf(self.channels, montage_split[:, 0].tolist())
        indices_second = get_pos_edf(self.channels, montage_split[:, 1].tolist())
        # montage
        self.data = self.data[indices_first, :] - self.data[indices_second, :]
        self.channels = montage
        return self

    def bandpass_filter(self, min_freq, max_freq, *, order=4):
        """Apply a bandpass filter to the data. Uses a Butterworth filter.

        Args:
            min_freq (float): cut-off frequency high-pass filter
            max_freq (float): cut-off frequency low-pass filter
            order (int, optional): Order of the butterworth filter. Defaults to 4.

        Returns:
            self: filter EEG object
        """

        self.data = ff.bandpass_filter(
            self.data, self.Fs, min_freq, max_freq, axis=1, order=order
        )
        return self

    # --------- save ------------
    def save(self, filename, saveas=".mat", *, eeg_file=None):
        """Save EEG object to a specific file type

        Args:
            filename (str): name of file to save to.
            saveas (str, optional): type of file to save to ".hdf5", ".h5" or ".mat". Defaults to ".mat".
            eeg_file (str, optional): name of original EEG file. If None, the end of the filename of the EEG object is used. Defaults to None.
        """

        if saveas == ".mat":
            savedict = dict()
            savedict["data"] = self.data
            savedict["fs"] = self.Fs
            savedict["channels"] = self.channels
            savedict["annotations"] = self.annotations
            savedict["labels"] = self.get_labels()
            savemat(filename, savedict, oned_as="column")
        elif saveas == ".hdf5" or saveas == ".h5":
            if not eeg_file:
                eeg_file = Path(self.filename).stem
            if ".hdf5" not in filename:
                filename = filename + saveas

            with h5py.File(filename, "a") as f:
                grp = f.require_group(eeg_file)
                if "eeg" in grp:
                    del grp["eeg"]

                eeg_data = grp.create_dataset("eeg", data=self.data.astype(np.float32))
                eeg_data.attrs["channels"] = self.channels
                eeg_data.attrs["fs"] = self.Fs
                if "labels" in grp:
                    del grp["labels"]

                ann_data = grp.create_dataset(
                    "labels", data=self.get_labels().astype(np.int8)
                )

        return

    # ----------- visualize ----------------

    def plot(self):
        """Plot the EEG data.

        Raises:
            Exception: if channels are anything other than a list or numpy array.
        """
        if isinstance(self.channels, np.ndarray):
            channels = self.channels.tolist()
        elif isinstance(self.channels, list):
            channels = self.channels
        else:
            raise Exception("Channels must be instance of list or ndarray")
        info = mne.create_info(channels, self.Fs, ch_types="eeg")
        raw = mne.io.RawArray(self.data * 1.0e-6, info)
        raw.plot()

        return

    def show(self):
        """same as plot"""
        self.plot()
        return

    # ----------- "get" methods ----------------

    def get_file_duration(self):
        try:
            f = pyedflib.EdfReader(self.filename)
        except IOError:
            print("Failed to open %s" % self.filename)
            raise
        self._file_duration = f.getFileDuration()
        f.close()
        return self._file_duration

    def get_sampling_frequency(self):
        try:
            f = pyedflib.EdfReader(self.filename)
        except IOError:
            print("Failed to open %s" % self.filename)
            raise
        self.Fs = f.getSampleFrequencies()
        f.close()
        if np.all(self.Fs == self.Fs[0]):
            self.Fs = self.Fs[0]
        return self.Fs

    def get_channels(self):
        try:
            f = pyedflib.EdfReader(self.filename)
        except IOError:
            print("Failed to open %s" % self.filename)
            raise
        channels = f.getSignalLabels()
        self._orig_channels = channels
        f.close()
        return channels

    def get_time(self):
        """Get a time vector for the EEG data

        Returns:
            ndarray: Array with time (in s) for each datapoint starting from 0 s.
        """
        ts = 1 / self.Fs
        n_samp = self.data.shape[1]
        timevec = np.arange(0, n_samp * ts, ts)
        assert len(timevec) == n_samp
        return timevec

    def get_labels(self):
        """Get the labels corresponding to the annotations. If annotation is "bckg" then label is -1. If annotation is any type of seizure then label is 1.

        Returns:
            list: list of the labels for each time point.
        """

        if self.annotations is None:
            return None

        timevec = self.get_time()
        labels = -1 * np.ones_like(timevec)

        seiz_df = self.annotations[self.annotations["annotation"] != "bckg"].copy()

        for idx, row in seiz_df.iterrows():
            idx_labels = np.where(
                np.logical_and(timevec >= row["start_time"], timevec < row["stop_time"])
            )
            labels[idx_labels] = 1

        return labels


def get_pos_edf(label_list: list, target_labels: list):
    """Get position of channels in edf file.

    Args:
        label_list (list): List of labels/channels
        target_labels (list): List of target labels (labels to extract)

    Raises:
        Exception: Failed to find label

    Returns:
        indices
    """
    indices = []  # indices of the target labels
    for lbl in target_labels:
        index = [
            i for i, elem in enumerate(label_list) if lbl.casefold() in elem.casefold()
        ]
        if len(index) == 0:
            raise Exception("Failed to find label %s" % lbl)
        else:
            indices.append(index[0])

    return indices


if __name__ == "__main__":
    from seizure_data_processing.config import MIT_CHB_DIR

    # file = (
    #     config.TUSZ_DIR
    #     + r"edf\train\01_tcp_ar\000\00000077\s003_2010_01_21\00000077_s003_t000.edf"
    # )
    # eeg_file = EEG(file)
    file = MIT_CHB_DIR + "chb01/chb01_03.edf"
    channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3"]
    eeg = EEG(file, channels)
