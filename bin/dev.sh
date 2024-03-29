#!/bin/bash

# Shell tmux script to start application

# create the session to be used
tmux new-session -d -s bmds

# split the windows
tmux split-window -h
tmux select-pane -t 1

# run commands
tmux send-keys -t 0 "source venv/bin/activate" enter
tmux send-keys -t 1 "source venv/bin/activate && ipython" enter

# attach to shell
tmux select-pane -t 0
tmux attach-session
