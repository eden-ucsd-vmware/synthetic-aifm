SHENANGO_PATH=../../../eden
ROOT_PATH=$(SHENANGO_PATH)
include $(SHENANGO_PATH)/shared.mk

librt_libs = $(SHENANGO_PATH)/bindings/cc/librt++.a
INC += -I$(SHENANGO_PATH)/bindings/cc -Iinc -I$(SHENANGO_PATH)/ksched \
	-Isnappy/build -Isnappy

main_src = main.cpp
main_obj = $(main_src:.cpp=.o)

lib_src = src/local_concurrent_hopscotch.cpp src/slab.cpp src/helpers.cpp src/MurmurHash3.cpp
lib_obj = $(lib_src:.cpp=.o)

src = $(main_src) $(lib_src)
obj = $(src:.cpp=.o)
dep = $(obj:.o=.d)

CXXFLAGS := $(filter-out -std=gnu++17,$(CXXFLAGS))
override CXXFLAGS += -std=gnu++2a -fconcepts -Wno-unused-function -mcmodel=medium
override RUNTIME_LIBS += -lcryptopp -Lsnappy/build -lsnappy 

ifneq ($(EDEN_SHIM),)
override RUNTIME_LIBS += -ljemalloc
endif

ifneq ($(PROVIDED_CXXFLAGS),)
override CXXFLAGS += $(PROVIDED_CXXFLAGS)
endif

#must be first
all: main

main: $(main_obj) $(librt_libs) $(RUNTIME_DEPS) $(main_obj) $(lib_obj)
	$(LDXX) -o $@ $(LDFLAGS) $(main_obj) $(lib_obj) $(librt_libs) $(RUNTIME_LIBS)

ifneq ($(MAKECMDGOALS),clean)
-include $(dep)   # include all dep files in the makefile
endif

#rule to generate a dep file by using the C preprocessor
#(see man cpp for details on the - MM and - MT options)
%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o $(dep) main src/*.o
