PYTHON ?= python3
PYTHON_CONFIG ?= $(shell $(PYTHON)-config --prefix 2>/dev/null && echo $(PYTHON)-config || echo python3-config)

CXXFLAGS += -O3 -march=native -Wall -shared -std=c++17 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell $(PYTHON) -m pybind11 --includes)
LIBNAME = bestfit_packer
LIBEXT = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# macOS: defer symbol resolution to load time (same as Linux behaviour)
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    LDFLAGS += -undefined dynamic_lookup
endif

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $< -o $@

clean:
	rm -f $(LIBNAME)*.so $(LIBNAME)*.pyd
