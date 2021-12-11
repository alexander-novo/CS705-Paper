# Upgraded to c++17 to support the Filesystem library
CXXFLAGS     = -std=c++17 -O3 -g
OBJDIR       = obj
DEPDIR       = $(OBJDIR)/.deps
# Flags which, when added to gcc/g++, will auto-generate dependency files
DEPFLAGS     = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

# Function which takes a list of words and returns a list of unique words in that list
# https://stackoverflow.com/questions/16144115/makefile-remove-duplicate-words-without-sorting
uniq         = $(if $1,$(firstword $1) $(call uniq,$(filter-out $(firstword $1),$1)))

# Source files - add more to auto-compile into .o files
SOURCES      = main.cpp nn-train.cpp common.cpp
INCLUDES     = -Iinclude
# Executable targets - add more to auto-make in default 'all' target
EXEC         = main nn-train

SOURCEDIRS   = $(call uniq, $(dir $(SOURCES)))
OBJDIRS      = $(addprefix $(OBJDIR)/, $(SOURCEDIRS))
DEPDIRS      = $(addprefix $(DEPDIR)/, $(SOURCEDIRS))
DEPFILES     = $(SOURCES:%.cpp=$(DEPDIR)/%.d)

.PHONY: all exec clean report required
.SECONDARY:

# By default, make all executable targets and the outputs required for the homework
all: exec
exec: $(EXEC)

# Executable Targets
main: $(OBJDIR)/main.o $(OBJDIR)/common.o
	$(CXX) $(CXXFLAGS) $^ -o $@ /usr/local/lib/libseal-3.7.a

nn-train: $(OBJDIR)/nn-train.o $(OBJDIR)/common.o
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(OBJDIR)
	rm -f $(EXEC)

# Auto-Build .cpp files into .o
$(OBJDIR)/%.o: %.cpp
$(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d | $(DEPDIRS) $(OBJDIRS)
	$(CXX) $(DEPFLAGS) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

# Make generated directories
$(DEPDIRS) $(OBJDIRS): ; @mkdir -p $@
$(DEPFILES):
include $(wildcard $(DEPFILES))