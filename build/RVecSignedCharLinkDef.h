// RVecSignedCharLinkDef.h
#include <vector>
#include "ROOT/RVec.hxx"
#pragma link C++ class ROOT::VecOps::RVec<signed char>+;
#pragma link C++ class ROOT::VecOps::RVec<unsigned char>+;
#pragma link C++ class ROOT::VecOps::RVec<int>+;
#pragma link C++ class ROOT::VecOps::RVec<float>+;
#pragma link C++ class ROOT::VecOps::RVec<bool>+;

// Add std::vector instantiations
#pragma link C++ class std::vector<signed char>+;
#pragma link C++ class std::vector<unsigned char>+;
#pragma link C++ class std::vector<int>+;
#pragma link C++ class std::vector<float>+;
#pragma link C++ class std::vector<bool>+;