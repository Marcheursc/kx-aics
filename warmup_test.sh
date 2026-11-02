#!/bin/bash
PATH=/root/miniconda3/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/projects/kx-aics
/root/miniconda3/bin/python -c "from src.chain import ModelPool; ModelPool(); print('OK')"
