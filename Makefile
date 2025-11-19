CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra
TARGET = polymer_sim
SOURCES = P2.cpp
HEADERS = Polymer.h
PYTHON_EXEC = C:/Users/Ale/miniconda3/envs/snakes/python.exe

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
	$(PYTHON_EXEC) plots.py

help:
	@echo "Available targets:"
	@echo "  make          - Compile the program"
	@echo "  make run      - Compile and run simulation"
	@echo "  make plot     - Run simulation and generate plots"
	@echo "  make clean    - Remove generated files"
	@echo "  make help     - Show this help message"