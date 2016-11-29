#!/bin/bash

# Shell tmux script to start application

# create the session to be used
tmux new-session -d -s bmds

# split the window
tmux split-window -h

# run commands
tmux send-keys -t 0 "workon bmds" enter
tmux send-keys -t 1 "workon bmds && make servedocs" enter

# attach to shell
tmux select-pane -t 0
tmux attach-session
