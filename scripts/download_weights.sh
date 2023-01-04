#!/bin/bash

wget https://www.dropbox.com/s/2co64p3y6d1nfex/weights_dc.tar.gz
mkdir -p weights
tar -xf weights_dc.tar.gz -C weights --strip-components=1
rm weights_dc.tar.gz
