#!/bin/bash

# 1. Doi 10 giay de dam bao Driver Camera va Desktop da khoi dong xong
sleep 5

# 2. Thiet lap bien moi truong man hinh (bat buoc cho app GUI)
export DISPLAY=:0

# 3. Di chuyen vao thu muc chua code
cd /home/vdas/Desktop/projects/VSLearner

# 4. Chay ung dung bang Python trong moi truong ao (pjs-env)
# Luu y: Duong dan nay phai chinh xac tuyet doi
#/home/vdas/Desktop/projects/pjs-env/bin/python gui_app.py &

sleep 3

xdotool key F11

wait
