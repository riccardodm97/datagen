import numpy as np
from pipelime.sequences.writers.filesystem import UnderfolderWriter
from pipelime.sequences.samples import PlainSample, SamplesSequence



IMAGE_KEY = 'image'
DEPTH_KEY = 'depth'
PROVA_KEY = 'light'
POSE_KEY = 'pose'
CAMERA_KEY = 'camera'
BBOX_KEY = 'bbox'

EXTENSIONS = {
        IMAGE_KEY: "png",
        DEPTH_KEY: "npy",
        PROVA_KEY: "txt",
    }

out_underfolder_path = '/home/eyecan/dev/real_relight/data/datasets/PROVA'

writer = UnderfolderWriter(
        folder= out_underfolder_path,
        root_files_keys=[PROVA_KEY],
        extensions_map=EXTENSIONS,
    )
samples = []
prova = np.random.randint(0,9,(4,4,2))
data = {
 PROVA_KEY: prova[0]
}

sample = PlainSample(data=data, id=0) 
samples.append(sample)

writer(SamplesSequence(samples))