all: open

TARGET = segment-tree-spikes.gif
SOURCE = illustrate.py

.PHONY: all format open

$(TARGET): $(SOURCE)
	pip3 install -r requirements.txt
	python3 $<

format: $(SOURCE)
	pip3 install black
	black $<

open: $(TARGET)
	if [ $$(uname -s) = 'Darwin' ]; then open $<; else xdg-open $< > /dev/null 2>&1 & fi
