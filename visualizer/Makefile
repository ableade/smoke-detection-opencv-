SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

# Use the name of the current directory as the name of the executable: 
all:  colorhistvisualizer


DEPDIR = deps
$(shell mkdir -p $(DEPDIR) >/dev/null)

CXXFLAGS += -std=c++11
CXXFLAGS += -O3
CXXFLAGS += -g
CXXFLAGS += -Wall -Wextra -pedantic
CXXFLAGS += -Weffc++
CXXFLAGS += -Werror=reorder
CXXFLAGS += -Werror=return-type

OPENCV_LINK_FLAGS += -lopencv_core
OPENCV_LINK_FLAGS += -lopencv_highgui
OPENCV_LINK_FLAGS += -lopencv_imgcodecs
OPENCV_LINK_FLAGS += -lopencv_imgproc
OPENCV_LINK_FLAGS += -lopencv_videoio
OPENCV_LINK_FLAGS += -lopencv_ml

LFLAGS += -lboost_filesystem
LFLAGS += -lboost_system
LFLAGS += $(OPENCV_LINK_FLAGS)

colorhistvisualizer : hist.cpp
	$(CXX) -o $@ $< $(LFLAGS)

clean:
	rm -f $(OBJECTS) $(addprefix $(DEPDIR)/, $(DEPS)) colorhistvisualizer