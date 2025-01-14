all: open

TARGET = segment-tree-spikes.gif

.PHONY: all open

$(TARGET): animate.py
	pip3 install -r requirements.txt
	python3 animate.py

open: $(TARGET)
	if [ $$(uname -s) = 'Darwin' ]; then open $<; else xdg-open $< > /dev/null 2>&1 & fi
