include Makefile.inc

INC_FOLDER = src
SRC_FOLDER = src
OBJ_FOLDER = obj
SRC_EXE_FOLDER = $(SRC_FOLDER)/exe
SRC_TEST_FOLDER = $(SRC_FOLDER)/test
OBJ_EXE_FOLDER = $(OBJ_FOLDER)/exe
OBJ_TEST_FOLDER = $(OBJ_FOLDER)/test
EXE_FOLDER = exe
TEST_FOLDER = test

INC_FILES = $(wildcard $(INC_FOLDER)/*.h)

SRC_FILES = $(wildcard $(SRC_FOLDER)/*.cpp)
SRC_EXE_FILES = $(wildcard $(SRC_EXE_FOLDER)/*.cpp) 
SRC_TEST_FILES = $(wildcard $(SRC_TEST_FOLDER)/*.cpp) 

OBJ_FILES = $(addprefix $(OBJ_FOLDER)/,$(notdir $(SRC_FILES:.cpp=.o)))
OBJ_EXE_FILES = $(addprefix $(OBJ_EXE_FOLDER)/,$(notdir $(SRC_EXE_FILES:.cpp=.o)))
OBJ_TEST_FILES = $(addprefix $(OBJ_TEST_FOLDER)/,$(notdir $(SRC_TEST_FILES:.cpp=.o)))
ALL_OBJ_FILES = $(OBJ_FILES) $(OBJ_EXE_FILES) $(OBJ_TEST_FILES)

EXE_FILES = $(addprefix $(EXE_FOLDER)/,$(notdir $(SRC_EXE_FILES:.cpp=)))
TEST_FILES = $(addprefix $(TEST_FOLDER)/,$(notdir $(SRC_TEST_FILES:.cpp=)))

CPP_FLAGS_RELEASE = -O3 -w
CPP_FLAGS_DEBUG = -O0 -g -Wall -Wconversion
CPP_FLAGS += $(CPP_FLAGS_$(BUILD))
CPP_FLAGS += -fpic -std=c++11 -I ${INC_FOLDER}

all: $(EXE_FILES) $(TEST_FILES) $(ALL_OBJ_FILES) 

dirguard=@mkdir -p $(@D)

# Build object files
$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.cpp $(INC_FILES)
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
	rm -f mpiout*
	rm -f mpitestout*
