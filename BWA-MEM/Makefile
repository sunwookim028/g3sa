# Compilers and flags
INCLUDES = -Isrc -Iext/zlib-1.3.1 -Iext/safestringlib/include -Iext/bwa-mem2/include
CXX = g++
CXXFLAGS = -Wall -std=c++11 $(INCLUDES) -MMD -MP -O3 #-lineinfo -g #-O0
CC = gcc
CFLAGS = -Wall -Wno-unused-function -O3 $(INCLUDES) -lm -DUSE_MALLOC_WRAPPERS -MMD -MP
NVCC = /usr/local/cuda-12.1/bin/nvcc
CU_ARCH = sm_86
CU_COMPUTE_ARCH = $(subst sm,compute,$(CU_ARCH))

__NVFLAGS = -ccbin /usr/bin/g++-11 --gpu-architecture=$(CU_COMPUTE_ARCH) --gpu-code=$(CU_ARCH) --default-stream per-thread $(INCLUDES) -Xptxas -O4 -Xcompiler -O4
_NVFLAGS = -ccbin /usr/bin/g++-11 --gpu-architecture=$(CU_COMPUTE_ARCH) --gpu-code=$(CU_ARCH) --default-stream per-thread $(INCLUDES) -MMD -MP -dc
NVFLAGS = $(_NVFLAGS) -Xptxas -O4 -Xcompiler -O4  -lineinfo
NVFLAGS_DEBUG = $(_NVFLAGS) -G -g

NVFLAGS=$(NVFLAGS_DEBUG)


# Linker flags
LINKFLAGS = -Lext/zlib-1.3.1 -Lext/safestringlib -L/usr/local/cuda/lib64 -Lext/bwa-mem2
LIBS = -lz -lcudart -lcudadevrt -lsafestring -lreadindexele


# Archiver and flags
AR = ar
ARFLAGS = -csru


# Target executable
TARGET1 = hasher
TARGET2 = g3
TARGET2_DEBUG = g3.debug


# Sources, objects and dependencies files
CPP_SOURCES = $(wildcard src/*.cpp)
C_SOURCES = $(wildcard src/*.c)
CU_SOURCES = $(wildcard src/*.cu)

CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CPP_OBJECTS_TARGET1 = src/buildIndex.o \
		      src/hashKMer.o \
		      src/loadKMerIndex.o
CPP_OBJECTS_TARGET2 = src/loadKMerIndex.o \
		      src/timer.o \
		      src/fastmap.o \
		      src/main.o \
		      src/FMI_wrapper.o \
		      src/FMI_search.o \
		      src/utils.o \
		      src/kopen.o \
		      src/memcpy_bwamem.o \
		      src/pipeline.o

C_OBJECTS = $(C_SOURCES:.c=.o)

C_OBJECTS_TARGET2 = src/malloc_wrap.o \
		    src/bwt.o \


CU_DEBUG_OBJECTS = $(CU_SOURCES:.cu=.debug.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

CU_DEBUG_OBJECTS_LINKER = src/gpu_link.debug.o
CU_OBJECTS_LINKER = src/gpu_link.o

DEPS_FILE = .depend



# Compile rules

-include $(DEPS_FILE)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<
%.debug.o: %.cu
	$(NVCC) $(NVFLAGS_DEBUG) -c -o $@ $<

# Default rule
all: depend $(TARGET1) $(TARGET2)


# Linking rule
$(TARGET1): $(CPP_OBJECTS_TARGET1)
	$(CXX) $(LINKFLAGS) -o $@ $^ -lz

$(TARGET2): $(CPP_OBJECTS_TARGET2) $(C_OBJECTS_TARGET2) $(CU_OBJECTS_LINKER) $(CU_OBJECTS) 
	$(CXX) $(LINKFLAGS) -L. -o $@ $^ $(LIBS) 

$(TARGET2_DEBUG): $(CU_DEBUG_OBJECTS_LINKER) $(CU_DEBUG_OBJECTS) $(CPP_OBJECTS_TARGET2)
	$(CXX) $(LINKFLAGS) -L. -o $@ $^ -lz -lcudart -lcudadevrt -lsafestring


$(CU_OBJECTS_LINKER): $(CU_OBJECTS)
	$(NVCC) $(__NVFLAGS) -dlink $^ --output-file $@

$(CU_DEBUG_OBJECTS_LINKER): $(CU_DEBUG_OBJECTS)
	$(NVCC) $(__NVFLAGS) -dlink $^ --output-file $@


#########################################################################################
# Make rules
#########################################################################################
EXE = $(TARGET2)
#REF_EXE = bwa-mem2
#REF_TARGETS = build run

build: $(EXE)

#ref:
#	$(MAKE) -C ../bwa-mem2 $(REF_TARGETS) REF_EXE=$(REF_EXE)


# Clean rule
clean:
	rm *sam *out *err

cleanlarge:
	rm -f $(CU_DEBUG_OBJECTS_LINKER) $(CU_OBJECTS_LINKER) $(CU_DEBUG_OBJECTS) $(CU_OBJECTS) $(CPP_OBJECTS) $(C_OBJECTS) $(TARGET1) $(TARGET2) $(DEPS_FILE)
	#$(MAKE) -C ../bwa-mem2 clean
	#rm -f ../bwa-mem2/$(EVALSUBDIR)/*


# Depend rule
depend:
	@echo "Generating dependencies..."
	@$(CXX) -MM $(CPP_SOURCES) $(INCLUDES) > $(DEPS_FILE)
	@$(CC) -MM $(C_SOURCES) $(INCLUDES) >> $(DEPS_FILE)
	@$(NVCC) -MM $(CU_SOURCES) $(INCLUDES) >> $(DEPS_FILE)


# Phony label
.PHONY: all clean depend

###########################################################################

# For debugging this build system 
debug_make: $(CU_DEBUG_OBJECTS)
	@echo "Compiled cuda object files."

compile: $(CPP_OBJECTS)
