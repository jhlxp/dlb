.PNONY: build run clean debug plot

TARGET = all2all_simulator
DLB_TEST_TARGET = dlb_fast_comparison_test
CC = g++
CFLAGS = -O3 -std=c++11
TEST_DIR = $(shell pwd)
BUILD_DIR = $(shell pwd)/build
ALGO_DIR = $(shell pwd)/algorithm
SCHE_DIR = $(shell pwd)/scheduler
DLB_DIR = $(shell pwd)/dlb
DLB_OUTPUT_DIR = $(DLB_DIR)/outputs
PLOT_DIR := $(shell pwd)/plot

INCLUDES = -I$(shell pwd)
INCLUDES += -I$(ALGO_DIR)
INCLUDES += -I$(SCHE_DIR)

OBJS = $(BUILD_DIR)/matrix.o
OBJS += $(BUILD_DIR)/all2all.o
OBJS += $(BUILD_DIR)/test.o
OBJS += $(BUILD_DIR)/config.o
OBJS += $(BUILD_DIR)/local.o
OBJS += $(BUILD_DIR)/global.o
DLB_TEST_OBJS = $(BUILD_DIR)/dlb_scheduler.o


LD = -lpthread

SOURCES = main.cpp

build: main.cpp define.h $(BUILD_DIR)/matrix.o $(BUILD_DIR)/all2all.o $(BUILD_DIR)/test.o $(BUILD_DIR)/local.o $(BUILD_DIR)/global.o $(BUILD_DIR)/config.o
	@echo "Building FastAll2All"
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "creating build directory"; \
		mkdir $(BUILD_DIR);\
	fi
	$(CC) -g $(CFLAGS) $(INCLUDES) $(OBJS) $(SOURCES) -o $(BUILD_DIR)/$(TARGET) $(LD)
	@echo "Building successful"

test-dlb: $(BUILD_DIR)/$(DLB_TEST_TARGET)
	$(BUILD_DIR)/$(DLB_TEST_TARGET)

benchmark-dlb: $(BUILD_DIR)/$(DLB_TEST_TARGET)
	@mkdir -p $(DLB_OUTPUT_DIR)
	$(BUILD_DIR)/$(DLB_TEST_TARGET) --benchmark-csv $(DLB_OUTPUT_DIR)/fast_dlb_comparison.csv
	python3 $(DLB_DIR)/plot_fast_dlb_comparison.py $(DLB_OUTPUT_DIR)/fast_dlb_comparison.csv $(DLB_OUTPUT_DIR)

$(BUILD_DIR)/$(DLB_TEST_TARGET): $(DLB_DIR)/dlb_fast_comparison_test.cpp $(DLB_TEST_OBJS) $(BUILD_DIR)/matrix.o $(BUILD_DIR)/all2all.o $(BUILD_DIR)/config.o $(BUILD_DIR)/local.o
	$(CC) -g $(CFLAGS) $(INCLUDES) -I$(DLB_DIR) $(DLB_TEST_OBJS) $(BUILD_DIR)/matrix.o $(BUILD_DIR)/all2all.o $(BUILD_DIR)/config.o $(BUILD_DIR)/local.o $(DLB_DIR)/dlb_fast_comparison_test.cpp -o $(BUILD_DIR)/$(DLB_TEST_TARGET) $(LD)

$(BUILD_DIR)/dlb_scheduler.o: $(DLB_DIR)/dlb_scheduler.cpp $(DLB_DIR)/dlb_scheduler.h
	$(CC) $(CFLAGS) -I$(DLB_DIR) -c $(DLB_DIR)/dlb_scheduler.cpp -o $(BUILD_DIR)/dlb_scheduler.o

$(BUILD_DIR)/all2all.o: $(BUILD_DIR)/matrix.o ${ALGO_DIR}/all2all.cpp $(ALGO_DIR)/all2all.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c ${ALGO_DIR}/all2all.cpp -o $(BUILD_DIR)/all2all.o

$(BUILD_DIR)/matrix.o: ${ALGO_DIR}/matrix.cpp $(ALGO_DIR)/matrix.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c ${ALGO_DIR}/matrix.cpp -o $(BUILD_DIR)/matrix.o

$(BUILD_DIR)/test.o: $(BUILD_DIR)/all2all.o $(BUILD_DIR)/matrix.o test.cpp test.h define.h $(BUILD_DIR)/config.o $(BUILD_DIR)/local.o $(BUILD_DIR)/global.o
	$(CC) $(CFLAGS) $(INCLUDES) -c test.cpp -o $(BUILD_DIR)/test.o

$(BUILD_DIR)/config.o: $(SCHE_DIR)/config.cpp $(SCHE_DIR)/config.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SCHE_DIR)/config.cpp -o $(BUILD_DIR)/config.o

$(BUILD_DIR)/local.o: $(BUILD_DIR)/config.o $(SCHE_DIR)/local.cpp $(SCHE_DIR)/local.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SCHE_DIR)/local.cpp -o $(BUILD_DIR)/local.o

$(BUILD_DIR)/global.o: $(BUILD_DIR)/config.o  $(BUILD_DIR)/matrix.o $(BUILD_DIR)/all2all.o $(SCHE_DIR)/global.cpp $(SCHE_DIR)/global.h define.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SCHE_DIR)/global.cpp -o $(BUILD_DIR)/global.o

run:
	$(BUILD_DIR)/$(TARGET)

clean:
	@echo "Cleaning FastAll2All"
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

debug:
	sudo gdb --args $(BUILD_DIR)/$(TARGET)

plot: benchmark-dlb FORCE
	@if [ -d benchmark ]; then \
		cd $(PLOT_DIR) && python3 plot_server_gpu_number.py && python3 plot_intra_speed_skewness.py; \
	else \
		echo "Skipping legacy FAST plots: benchmark/ data is absent"; \
	fi

FORCE: ;
