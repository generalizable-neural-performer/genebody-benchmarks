# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import data.genebody as datamodel

def get_dataset(datadir, annotdir, subject, subsampletype=None):
    return datamodel.Dataset(
        datadir,
        annotdir,
        subject,
        keyfilter=["bg", "fixedcamimage", "camera", "image", "pixelcoords"],
        fixedcammean=100.,
        fixedcamstd=25.,
        imagemean=100.,
        imagestd=25.,
        subsampletype=subsampletype,
        subsamplesize=128)

def get_autoencoder(dataset):
    import models.neurvol_genebody as aemodel
    import models.encoders.mvconv as encoderlib
    import models.decoders.voxel1 as decoderlib
    import models.volsamplers.warpvoxel as volsamplerlib
    import models.colorcals.colorcal1 as colorcalib
    return aemodel.Autoencoder(
        dataset,
        encoderlib.Encoder(dataset.ninput),
        decoderlib.Decoder(globalwarp=True),
        volsamplerlib.VolSampler(),
        colorcalib.Colorcal([str(key) for key in dataset.get_allcameras()]),
        4. / 256)

### profiles
# A profile is instantiated by the training or evaluation scripts
# and controls how the dataset and autoencoder is created
class Train():
    batchsize=16
    maxiter=500000
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_dataset(self, datadir, annotdir, subject): 
        return get_dataset(datadir, annotdir, subject, subsampletype="random2")
    def get_optimizer(self, ae):
        import itertools
        import torch.optim
        lr = 0.0001
        aeparams = itertools.chain(
            [{"params": x} for x in ae.encoder.parameters()],
            [{"params": x} for x in ae.decoder.parameters()],
            [{"params": x} for x in ae.colorcal.parameters()])
        return torch.optim.Adam(aeparams, lr=lr, betas=(0.9, 0.999))
    def get_loss_weights(self):
        return {"irgbmse": 1.0, "kldiv": 0.001, "alphapr": 0.01, "tvl1": 0.01}

class ProgressWriter():
    def batch(self, iternum, itemnum, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        batchsize = kwargs["image"].size(0)
        if batchsize > 1:
            batchsize = int(np.sqrt(batchsize)) ** 2
            # st()
            for i in range(batchsize):
                row.append(
                    np.concatenate((
                            kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                            kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1)
                        )
                if len(row) == int(np.sqrt(batchsize)):
                    rows.append(np.concatenate(row, axis=1))
                    row = []
            imgout = np.concatenate(rows, axis=0)
        else:
            imgout = np.concatenate((
                        kwargs["irgbrec"][0].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                        kwargs["image"][0].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1)
        
        # outpath = os.path.dirname(__file__)
        outpath = kwargs["logdir"]
        Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, "prog_{:06}.jpg".format(iternum)))

class Progress():
    """Write out diagnostic images during training."""
    batchsize=16
    def get_ae_args(self): return dict(outputlist=["irgbrec"])
    def get_dataset(self, datadir, annotdir, subject): 
        return get_dataset(datadir, annotdir, subject)
    def get_writer(self): return ProgressWriter()

class Render():
    """Render model with training camera or from novel viewpoints.
    
    e.g., python render.py {configpath} Render --maxframes 128"""
    def __init__(self, subject, showtarget=False, showdiff=False, viewtemplate=False):
        self.subject = subject
        self.showtarget = showtarget
        self.viewtemplate = viewtemplate
        self.showdiff = showdiff
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_ae_args(self): return dict(outputlist=["irgbrec", "irgbsqerr"], viewtemplate=self.viewtemplate)
    def get_dataset(self, datadir, annotdir, subject):
        return get_dataset(datadir, annotdir, subject)
    def get_writer(self):
        import eval.writers.videowriter as writerlib
        return writerlib.Writer(
            os.path.join('logs', self.subject,
                "render{}.mp4".format(
                    "_template" if self.viewtemplate else "")),
            showtarget=self.showtarget,
            showdiff=self.showdiff)
