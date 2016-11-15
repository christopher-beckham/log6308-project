#!/bin/bash
HYBRID_BOTH_DEEPER_TIED_FIXED=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu2,allow_gc=True,floatX=float32 \
  python create_data.py
