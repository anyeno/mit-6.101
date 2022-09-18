# No Imports Allowed!


def backwards(sound):
    # 反转声音
    rate = sound['rate']
    samples = sound['samples']
    new_sam = []
    for i in range(len(samples)):
        new_sam.append(samples[len(samples) - i - 1])
    res = {
        'rate': rate,
        'samples': new_sam
        }
    return res


def mix(sound1, sound2, p):
    rate1 = sound1['rate']
    rate2 = sound2['rate']
    if rate1 != rate2:
        return None
    samples1 = sound1['samples']
    samples2 = sound2['samples']
    min_length = min(len(samples1), len(samples2))
    new_samples = []
    for i in range(min_length):
        new_samples.append(samples1[i] * p + samples2[i] * (1 - p))
    res = {'rate': rate1, 'samples': new_samples}

    return res

def echo(sound, num_echoes, delay, scale):
    rate = sound['rate']
    scales = []
    scale_in_scales = scale
    for i in range(num_echoes):
        scales.append(scale_in_scales)
        scale_in_scales = scale_in_scales * scale
    sample_delay = round(delay * rate)   #每个副本延迟多少个样本
    samples = sound['samples']
    new_samples = []
    for i in range(len(samples)):
        new_samples.append(samples[i])

    for i in range(num_echoes):
        to_be_added = []
        for j in range((i + 1) * sample_delay):
            to_be_added.append(0)
        for k in range(len(samples)):
            to_be_added.append(samples[k] * scales[i])
        for n in range(len(to_be_added)):
            if(n < len(new_samples)):
                new_samples[n] = new_samples[n] + to_be_added[n]
            else:
                new_samples.append(to_be_added[n])
    res = {'rate': rate, 'samples': new_samples}

    return res



def pan(sound):
    rate = sound['rate']
    left = sound['left']
    right = sound['right']
    new_left = left.copy()
    new_right = right.copy()
    N = len(left)
    scale = 0
    for i in range(N):
        scale = i / (N - 1)
        new_right[i] = right[i] * scale
    for i in range(N):
        scale = 1 - i / (N - 1)
        new_left[i] = left[i] * scale
    res = {
        'rate': rate,
        'left': new_left,
        'right': new_right
    }

    return res


def remove_vocals(sound):
    rate = sound['rate']
    left = sound['left']
    right = sound['right']
    samples = []
    for i in range(len(left)):
        samples.append(left[i] - right[i])
    res = {'rate': rate, 'samples': samples}

    return res


# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds

import io
import wave
import struct


def load_wav(filename, stereo=False):
    """
    Given the filename of a WAV file, load the data from that file and return a
    Python dictionary representing that sound
    //wav转换成python的字典
    """
    f = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = f.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    if stereo:
        left = []
        right = []
        for i in range(count):
            frame = f.readframes(1)
            if chan == 2:
                left.append(struct.unpack("<h", frame[:2])[0])
                right.append(struct.unpack("<h", frame[2:])[0])
            else:
                datum = struct.unpack("<h", frame)[0]
                left.append(datum)
                right.append(datum)

        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = []
        for i in range(count):
            frame = f.readframes(1)
            if chan == 2:
                left = struct.unpack("<h", frame[:2])[0]
                right = struct.unpack("<h", frame[2:])[0]
                samples.append((left + right) / 2)
            else:
                datum = struct.unpack("<h", frame)[0]
                samples.append(datum)

        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Given a dictionary representing a sound, and a filename, convert the given
    sound into WAV format and save it as a file with the given filename (which
    can then be opened by most audio players)
    python字典表示的声音转化成wav文件
    """
    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for l, r in zip(sound["left"], sound["right"]):
            l = int(max(-1, min(1, l)) * (2**15 - 1))
            r = int(max(-1, min(1, r)) * (2**15 - 1))
            out.append(l)
            out.append(r)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified aas
    # sounds/meow.wav, rather than just as meow.wav, to account for the sound
    # files being in a different directory than this file)
    sound = load_wav("sounds/再也看不见海.wav", True)
    new = remove_vocals(sound)
    write_wav(new, "new_sound.wav")

    # write_wav(backwards(meow), 'meow_reversed.wav')
