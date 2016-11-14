#!/bin/bash
HYBRID_USER_DEEPER_MASK=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,allow_gc=True,floatX=float32 \
  python create_data.py
