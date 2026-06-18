#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <string>
#include <unistd.h>

bool FindFile(const std::string& root, const std::string& filename, std::string* out) {
  DIR* dir = opendir(root.c_str());
  if (dir == nullptr) return false;
  while (dirent* entry = readdir(dir)) {
    if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) continue;
    const std::string path = root + "/" + entry->d_name;
    if (filename == entry->d_name) {
      *out = path;
      closedir(dir);
      return true;
    }
    if (entry->d_type == DT_DIR || entry->d_type == DT_LNK || entry->d_type == DT_UNKNOWN) {
      if (FindFile(path, filename, out)) {
        closedir(dir);
        return true;
      }
    }
  }
  closedir(dir);
  return false;
}

int main() {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  if (test_srcdir == nullptr) {
    std::cerr << "TEST_SRCDIR is not set\n";
    return 1;
  }
  std::string script;
  if (!FindFile(test_srcdir, "python_binding_test.py", &script)) {
    std::cerr << "could not find python_binding_test.py in runfiles\n";
    return 1;
  }
  const char* configured_python = std::getenv("CLQR_PYTHON");
  std::string python =
      configured_python == nullptr ? "/Users/joao/.pyenv/versions/3.13.3/bin/python3"
                                   : configured_python;
  if (access(python.c_str(), X_OK) != 0) python = "python3";
  const std::string command = "\"" + python + "\" \"" + script + "\"";
  const int status = std::system(command.c_str());
  if (status != 0) {
    std::cerr << "python binding smoke test failed with status " << status << "\n";
    return 1;
  }
  return 0;
}
