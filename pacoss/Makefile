include Makefile.inc

LIB_NAME = pacoss
INC_FOLDER = src
SRC_FOLDER = src
OBJ_FOLDER = obj
SRC_EXE_FOLDER = $(SRC_FOLDER)/exe
SRC_TEST_FOLDER = $(SRC_FOLDER)/test
OBJ_EXE_FOLDER = $(OBJ_FOLDER)/exe
OBJ_TEST_FOLDER = $(OBJ_FOLDER)/test
EXE_FOLDER = exe
TEST_FOLDER = test

INC_FILES = $(wildcard $(INC_FOLDER)/*.h $(TMPI_SRC_FOLDER)/*.h)

SRC_FILES = $(wildcard $(SRC_FOLDER)/*.cpp) $(wildcard $(TMPI_SRC_FOLDER)/*.cpp)
SRC_EXE_FILES = $(wildcard $(SRC_EXE_FOLDER)/*.cpp) 
SRC_TEST_FILES = $(wildcard $(SRC_TEST_FOLDER)/*.cpp) 

OBJ_FILES = $(addprefix $(OBJ_FOLDER)/,$(notdir $(SRC_FILES:.cpp=.o)))
OBJ_EXE_FILES = $(addprefix $(OBJ_EXE_FOLDER)/,$(notdir $(SRC_EXE_FILES:.cpp=.o)))
OBJ_TEST_FILES = $(addprefix $(OBJ_TEST_FOLDER)/,$(notdir $(SRC_TEST_FILES:.cpp=.o)))
ALL_OBJ_FILES = $(OBJ_FILES) $(OBJ_EXE_FILES) $(OBJ_TEST_FILES)

EXE_FILES = $(addprefix $(EXE_FOLDER)/,$(notdir $(SRC_EXE_FILES:.cpp=)))
TEST_FILES = $(addprefix $(TEST_FOLDER)/,$(notdir $(SRC_TEST_FILES:.cpp=)))

CPP_FLAGS_RELEASE = -O3
CPP_FLAGS_DEBUG = -O0 -g
CPP_FLAGS += $(CPP_FLAGS_$(BUILD)) -Wall -Wconversion -Werror
CPP_FLAGS += -fpic -std=c++11 -I ${INC_FOLDER} -I ${TMPI_SRC_FOLDER}
ifneq ("$(PATOH_FOLDER)", "")
	CPP_FLAGS += -DPACOSS_USE_PATOH -I$(PATOH_FOLDER)/inc
	LD_FLAGS += $(PATOH_FOLDER)/lib/libpatoh.a
endif

all: $(EXE_FILES) $(TEST_FILES) $(ALL_OBJ_FILES) liba

dirguard=@mkdir -p $(@D)

# Build object files
$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.cpp $(INC_FILES)
	$(dirguard)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

$(OBJ_FOLDER)/%.o: $(TMPI_SRC_FOLDER)/%.cpp $(INC_FILES)
	$(dirguard)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

$(OBJ_EXE_FOLDER)/%.o: $(SRC_EXE_FOLDER)/%.cpp $(INC_FILES)
	$(dirguard)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

$(OBJ_TEST_FOLDER)/%.o: $(SRC_TEST_FOLDER)/%.cpp $(INC_FILES)
	$(dirguard)
	$(CPP) $(CPP_FLAGS) -c -o $@ $<

# Build executables
$(EXE_FOLDER)/%: $(OBJ_FILES) $(OBJ_EXE_FOLDER)/%.o
	$(dirguard)
	$(CPP) $(CPP_FLAGS) -o $@ $^ $(LD_FLAGS)

# Build test routines
$(TEST_FOLDER)/%: $(OBJ_FILES) $(OBJ_TEST_FOLDER)/%.o
	$(dirguard)
	$(CPP) $(CPP_FLAGS) -o $@ $^ $(LD_FLAGS)

clean:
	rm -rf $(OBJ_FOLDER) $(EXE_FOLDER) $(TEST_FOLDER)
	rm -rf mpiout*
	rm -rf mpitestout*

print-%: ; @echo $* = $($*)

liba: $(OBJ_FILES)
	mkdir -p lib
	ar rcs lib/lib$(LIB_NAME).a $^

libso: $(OBJ_FILES)
	mkdir -p lib
	gcc -shared -o lib/lib$(LIB_NAME).so $^
