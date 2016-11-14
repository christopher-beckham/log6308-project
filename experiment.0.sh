#!/bin/bash
HYBRID_ITEM_ADAM_L2=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,allow_gc=True,floatX=float32 \
  python create_data.py
