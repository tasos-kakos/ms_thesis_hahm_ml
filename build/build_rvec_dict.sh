#!/usr/bin/env bash
set -euo pipefail


# Build script for ROOT dictionary
# Usage: activate your conda env with ROOT, then run ./build_rvec_dict.sh
# e.g. conda activate env_ms_hahm_thesis


LINKDEF="RVecSignedCharLinkDef.h"
DICT_SRC="RVecSignedDict.cxx"
LIBNAME="libRVecSignedChar.so"


echo "Using `which rootcling || true` and root-config:`root-config --version`"


# 1) generate dictionary C++ source
rootcling -f ${DICT_SRC} -c ${LINKDEF}


# 2) compile shared object
# Force c++17: some ROOT builds require modern standard
g++ -std=c++17 -fPIC -shared `root-config --cflags` ${DICT_SRC} -o ${LIBNAME} `root-config --libs`


echo "Built ${LIBNAME}"


echo "Quick test (load in Python with PyROOT):"
python - <<'PY'
import ROOT
import sys
lib='{}'
print('Loading', lib)
ROOT.gSystem.Load(lib)
print('Loaded libraries (excerpt):')
print(ROOT.gSystem.GetLibraries()[:200])
# Try a tiny Cling snippet that constructs the type (should not throw unknown-type)
try:
    ROOT.gInterpreter.ProcessLine('ROOT::VecOps::RVec<signed char> v; v.push_back(5); std::cout << (int)v[0] << std::endl;')
    print('Type instantiation test: OK')
except Exception as e:
    print('Type instantiation test: FAILED', e)
    sys.exit(1)
PY


echo "If the test printed 'Type instantiation test: OK', you can now load ${LIBNAME} from your script with:\n import ROOT\n ROOT.gSystem.Load('${LIBNAME}')\nthen create RDataFrame and call AsNumpy()."
