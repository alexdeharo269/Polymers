CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra
TARGET = polymer_sim
SOURCES = main.cpp
HEADERS = Polymer.h

.PHONY: all clean run plot help

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	@echo "Compiling $(TARGET)..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)
	@echo "Build complete!"

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET) *.csv *.png *.o
	@echo "Clean complete!"

run: $(TARGET)
	@echo "Running simulation..."
	./$(TARGET)

plot: run
	@echo "Generating plots..."
	python visualize.py

help:
	@echo "Available targets:"
	@echo "  make          - Compile the program"
	@echo "  make run      - Compile and run simulation"
	@echo "  make plot     - Run simulation and generate plots"
	@echo "  make clean    - Remove generated files"
	@echo "  make help     - Show this help message"