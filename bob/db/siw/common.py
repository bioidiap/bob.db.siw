import scipy.io.wavfile
from bob.db.base import read_annotation_file
from bob.io.base import load
from bob.io.video import reader
from bob.bio.video.utils import FrameSelector
from bob.bio.video.database import VideoBioFile
import numpy as np
import subprocess
import tempfile
import os
from . import SIW_FRAME_SHAPE

SITE_MAPPING = {
    '1': 'NTNU',
    '2': 'UIO',
    '3': 'MPH-FRA',
    '4': 'IDIAP',
    '6': 'MPH-IND',
}

DEVICE_MAPPING = {
    'p': 'iPhone',
    't': 'iPad',
}

MODALITY_MAPPING = {
    '1': 'face',
    '2': 'voice',
    '3': 'eye',
    '4': 'finger',
}


def read_audio(video_path):
    with tempfile.NamedTemporaryFile(suffix='.wav') as f:
        cmd = ['ffmpeg', '-v', 'quiet', '-i', video_path, '-y', '-vn', f.name]
        subprocess.call(cmd)
        f.seek(0)
        rate, signal = scipy.io.wavfile.read(f.name)
    return rate, signal


class Client(object):
    """A base class for SIW clients"""

    def __init__(self, site, id_in_site, gender, **kwargs):
        super(Client, self).__init__(**kwargs)
        self.institute = site
        self.id_in_site = id_in_site
        self.gender = gender

    @property
    def id(self):
        return '{}_{}'.format(self.institute, self.id_in_site)


def siw_file_metadata(path):
    """Returns the metadata associated with a SIW file

    All the video files are named as SubjectID_SensorID_TypeID_MediumID_SessionID.mov
    (or *.mp4). SubjectID ranges from 001 to 165. SensorID represents the capture
    device. TypeID represents the spoof type of the video. MediumID and SessionID record
    additional details of the video, shown in the Figure 2
    (http://cvlab.cse.msu.edu/spoof-in-the-wild-siw-face-anti-spoofing-database.html)

    Parameters
    ----------
    path : str
        The path of the SIW file.

    Returns
    -------
    client_id : str
    attack_type : str
    sensor_id : str
    type_id : str
    medium_id : str
    session_id : str
    """
    # For example:
    # path: Train/live/003/003-1-1-1-1.mov
    # path: Train/spoof/003/003-1-2-1-1.mov
    fldr, path = os.path.split(path)
    # live_spoof = os.path.split(os.path.split(fldr)[0])[1]
    path, extension = os.path.splitext(path)
    client_id, sensor_id, type_id, medium_id, session_id = path.split('-')
    attack_type = {"1": None, "2": "print", "3": "replay"}[type_id]
    if attack_type is not None:
        attack_type = f"{attack_type}/{medium_id}"
    return client_id, attack_type, sensor_id, type_id, medium_id, session_id

class SiwFile(object):
    """A base class for SIW bio files which can handle the metadata."""

    def __init__(self, **kwargs):
        super(SiwFile, self).__init__(**kwargs)
        (
            self.client, self.session, self.nrecording,
            self.device, self.modality
        ) = siw_file_metadata(self.path)


class SiwVideoFile(VideoBioFile, SiwFile):
    """A base class for SIW video files"""

    def swap(self, data):
        # rotate the video or image since SIW videos are not upright!
        return np.swapaxes(data, -2, -1)

    def load(self, directory=None, extension=None,
             frame_selector=FrameSelector(selection_style='all')):
        if extension is None:
            video_path = self.make_path(directory or self.original_directory,
                                        extension)
            for _ in range(100):
                try:
                    video = load(video_path)
                    break
                except RuntimeError:
                    pass
            video = self.swap(video)
            return frame_selector(video)
        else:
            return super(SiwVideoFile, self).load(
                directory, extension, frame_selector)

    @property
    def frames(self):
        """Yields the frames of the padfile one by one.

        Parameters
        ----------
        padfile : :any:`SiwVideoFile`
            The high-level pad file

        Yields
        ------
        :any:`numpy.array`
            A frame of the video. The size is (3, 1280, 720).
        """
        vfilename = self.make_path(directory=self.original_directory)
        video = reader(vfilename)
        for frame in video:
            yield self.swap(frame)

    @property
    def number_of_frames(self):
        """Returns the number of frames in a video file.

        Parameters
        ----------
        padfile : :any:`SiwVideoFile`
            The high-level pad file

        Returns
        -------
        int
            The number of frames.
        """
        vfilename = self.make_path(directory=self.original_directory)
        return reader(vfilename).number_of_frames

    @property
    def frame_shape(self):
        """Returns the size of each frame in this database.

        Returns
        -------
        (int, int, int)
            The (#Channels, Height, Width) which is (3, 1920, 1080).
        """
        return SIW_FRAME_SHAPE

    @property
    def annotations(self):
        """Returns the annotations of the current file

        Returns
        -------
        dict
            The annotations as a dictionary, e.g.:
            ``{'0': {'reye':(re_y,re_x), 'leye':(le_y,le_x)}, ...}``
        """
        return read_annotation_file(
            self.make_path(self.annotation_directory,
                           self.annotation_extension),
            self.annotation_type)


class SiwAudioFile(SiwVideoFile):
    """A base class that extracts audio from SIW video files"""

    def load(self, directory=None, extension=None):
        if extension is None:
            video_path = self.make_path(directory, extension)
            rate, audio = read_audio(video_path)
            return rate, np.cast['float'](audio)
        else:
            return super(SiwAudioFile, self).load(directory, extension)


class SiwVideoDatabase(object):
    """SiwVideoDatabase"""

    def frames(self, padfile):
        return padfile.frames

    def number_of_frames(self, padfile):
        return padfile.number_of_frames

    @property
    def frame_shape(self):
        return SIW_FRAME_SHAPE

    def update_files(self, files):
        for f in files:
            f.original_directory = self.original_directory
            f.annotation_directory = self.annotation_directory
            f.annotation_extension = self.annotation_extension
            f.annotation_type = self.annotation_type
        return files
