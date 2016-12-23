#!/bin/bash

# Shell tmux script to start application

# create the session to be used
tmux new-session -d -s bmds

# split the windows
tmux split-window -h
tmux select-pane -t 1
tmux split-window -v

# run commands
tmux send-keys -t 0 "workon bmds && bpython" enter
tmux send-keys -t 1 "workon bmds" enter
tmux send-keys -t 2 "workon bmds && make servedocs" enter

# attach to shell
tmux select-pane -t 0
tmux attach-session
